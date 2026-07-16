# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig
from transformers.integrations.fp_quant import adapt_fp_quant_config
from transformers.quantizers.quantizer_fp_quant import FPQuantHfQuantizer
from transformers.testing_utils import (
    backend_empty_cache,
    require_accelerate,
    require_fp_quant,
    require_qutlass,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
    torch_device,
)


class FPQuantConfigTest(unittest.TestCase):
    @staticmethod
    def schema_v1_dict(forward_dtype="int4"):
        format_value = {
            "int4": "int",
            "nvint4": "nvint",
            "mxint4": "mxint",
            "nvfp4": "nvfp",
            "mxfp4": "mxfp",
        }[forward_dtype]
        group_size = {"int4": None, "nvint4": 16, "mxint4": 32, "nvfp4": 16, "mxfp4": 32}[forward_dtype]
        granularity = "channel" if forward_dtype == "int4" else "group"
        tensor_contract = {
            "format": format_value,
            "bits": 4,
            "mapping": "signed" if "int" in forward_dtype else None,
            "granularity": granularity,
            "group_size": group_size,
            "scale_precision": {
                "int4": "fp16",
                "nvint4": "e4m3",
                "mxint4": "e8m0",
                "nvfp4": "e4m3",
                "mxfp4": "e8m0",
            }[forward_dtype],
            "observer": "minmax",
        }
        block_size = 16 if forward_dtype == "nvint4" else 32
        return {
            "quant_method": "fp_quant",
            "schema_version": 1,
            "runtime_contract_version": 1,
            "minimum_fp_quant_version": "0.3.2",
            "minimum_transformers_version": "4.57.6",
            "storage_mode": "pseudoquant",
            "quantization_spec": {
                "adapter": "llm",
                "operator": "linear",
                "operation": "rtn",
                "weight": dict(tensor_contract),
                "activation": dict(tensor_contract),
                "transform": "identity",
                "transform_block_size": None,
                "activation_order": "original",
                "gptq_block_size": 128,
                "rel_damp": 0.01,
            },
            "forward_dtype": forward_dtype,
            "backward_dtype": "bf16",
            "forward_method": "abs_max",
            "hadamard_group_size": block_size,
            "transform_init": "identity",
            "modules_to_not_convert": ["lm_head"],
            "store_master_weights": False,
            "pseudoquantization": True,
        }

    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = FPQuantConfig()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"modules_to_not_convert": ["embed_tokens", "lm_head"], "quant_method": "fp_quant"}
        quantization_config = FPQuantConfig.from_dict(dict)

        self.assertEqual(dict["modules_to_not_convert"], quantization_config.modules_to_not_convert)
        self.assertEqual(dict["quant_method"], quantization_config.quant_method)

    def test_schema_v1_round_trip_is_lossless(self):
        for forward_dtype in ["int4", "nvint4", "mxint4", "nvfp4", "mxfp4"]:
            with self.subTest(forward_dtype=forward_dtype):
                payload = self.schema_v1_dict(forward_dtype)

                quantization_config = FPQuantConfig.from_dict(payload)
                serialized = quantization_config.to_dict()

                for key, value in payload.items():
                    self.assertEqual(value, serialized[key])

    @require_fp_quant
    def test_schema_v1_adapter_is_lossless(self):
        for forward_dtype in ["int4", "nvint4", "mxint4", "nvfp4", "mxfp4"]:
            with self.subTest(forward_dtype=forward_dtype):
                config = FPQuantConfig.from_dict(self.schema_v1_dict(forward_dtype))

                local = adapt_fp_quant_config(config)

                self.assertEqual(local.schema_version, config.schema_version)
                self.assertEqual(local.runtime_contract_version, config.runtime_contract_version)
                self.assertEqual(local.storage_mode, config.storage_mode)
                self.assertEqual(local.quantization_spec, config.quantization_spec)
                self.assertEqual(local.forward_dtype.value, config.forward_dtype)

    def test_schema_v1_rejects_unsupported_versions_and_real_int(self):
        payload = self.schema_v1_dict()
        payload["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema_version 2"):
            FPQuantConfig.from_dict(payload)

        payload = self.schema_v1_dict()
        payload["storage_mode"] = "realquant"
        payload["pseudoquantization"] = False
        with self.assertRaisesRegex(ValueError, "only pseudoquantization"):
            FPQuantConfig.from_dict(payload)


@require_fp_quant
@require_accelerate
class FPQuantSchemaV1LoaderTest(unittest.TestCase):
    def setUp(self):
        self.config = FPQuantConfig.from_dict(FPQuantConfigTest.schema_v1_dict())
        self.quantizer = FPQuantHfQuantizer(self.config, pre_quantized=True)

    def test_int_pseudoquant_environment_allows_cpu_and_rejects_disk(self):
        with patch("transformers.quantizers.quantizer_fp_quant.torch.cuda.is_available", return_value=False):
            self.quantizer.validate_environment(device_map={"": "cpu"})
            with self.assertRaisesRegex(ValueError, "does not support disk"):
                self.quantizer.validate_environment(device_map={"": "disk"})

    def test_float_pseudoquant_environment_still_requires_gpu(self):
        config = FPQuantConfig(forward_dtype="mxfp4", pseudoquantization=True)
        quantizer = FPQuantHfQuantizer(config, pre_quantized=False)

        with (
            patch("transformers.quantizers.quantizer_fp_quant.torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(NotImplementedError, "requires GPU"),
        ):
            quantizer.validate_environment(device_map={"": "cpu"})

    def test_int_module_replacement_loads_baked_dqweight(self):
        model = nn.Module()
        model.proj = nn.Linear(32, 2, bias=True, dtype=torch.bfloat16)
        model.config = SimpleNamespace()

        self.quantizer._process_model_before_weight_loading(model)
        dqweight = torch.linspace(-1, 1, 64, dtype=torch.bfloat16).reshape(2, 32)
        self.quantizer.create_quantized_param(
            model,
            dqweight,
            "proj.dqweight",
            torch.device("cpu"),
        )
        module = model.proj
        module.bias = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16), requires_grad=False)
        module.forward_hadamard_matrix = torch.eye(32, dtype=torch.bfloat16)
        module.backward_hadamard_matrix = torch.eye(32, dtype=torch.bfloat16)
        module.weight_global_scale = torch.tensor(1.0)
        module.act_global_scale = torch.tensor(1.0)

        self.assertTrue(module._has_prequantized_dqweight)
        self.assertFalse(module.dqweight.requires_grad)
        self.assertEqual(module(torch.ones(1, 32, dtype=torch.bfloat16)).shape, (1, 2))
        self.assertEqual(
            self.quantizer.update_missing_keys(
                model,
                ["proj.weight", "proj.bias", "proj.forward_hadamard_matrix"],
                "",
            ),
            ["proj.bias", "proj.forward_hadamard_matrix"],
        )


@slow
@require_torch_gpu
@require_fp_quant
@require_accelerate
class FPQuantBaseTest(unittest.TestCase):
    model_name = "unsloth/Llama-3.2-1B"

    input_text = "1 2 3 4"
    max_new_tokens = 4

    EXPECTED_OUTPUT = "1 2 3 4 5 6"

    device_map = "cuda"

    @classmethod
    def getQuantizationConfig(cls):
        unittest.skip("Subclass must implement this method")

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """

        quantization_config = cls.getQuantizationConfig()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, quantization_config=quantization_config
        )

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUs
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)
        quantization_config = FPQuantConfig()
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=quantization_config
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_save_pretrained_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map="auto")
            self.assertTrue(set(model.hf_device_map.values()) == {0, 1})

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)


class FPQuantMXFP4PseudoquantTest(FPQuantBaseTest):
    @classmethod
    def getQuantizationConfig(cls):
        return FPQuantConfig(forward_dtype="mxfp4", pseudoquantization=True)


class FPQuantNVFP4PseudoquantTest(FPQuantBaseTest):
    @classmethod
    def getQuantizationConfig(cls):
        return FPQuantConfig(forward_dtype="nvfp4", pseudoquantization=True)


@require_qutlass
class FPQuantMXFP4Test(FPQuantBaseTest):
    @classmethod
    def getQuantizationConfig(cls):
        return FPQuantConfig(forward_dtype="nvfp4", pseudoquantization=False)


@require_qutlass
class FPQuantMXFP4GS128Test(FPQuantBaseTest):
    @classmethod
    def getQuantizationConfig(cls):
        return FPQuantConfig(forward_dtype="nvfp4", pseudoquantization=False, hadamard_group_size=128)


@require_qutlass
class FPQuantNVFP4GS128Test(FPQuantBaseTest):
    @classmethod
    def getQuantizationConfig(cls):
        return FPQuantConfig(forward_dtype="nvfp4", pseudoquantization=False, hadamard_group_size=128)
