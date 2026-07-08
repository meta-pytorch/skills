// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>

#include <ATen/ATen.h>

#include <string>
#include <tuple>

// Blockwise int8 quantization: returns (quantized values, per-block scales).
std::tuple<at::Tensor, at::Tensor> quantize_per_block(
    const at::Tensor& input,
    int block_size,
    float eps) {
  TORCH_CHECK(input.numel() % block_size == 0,
              "input numel must be divisible by block_size");
  auto blocks = input.contiguous().reshape({-1, block_size});
  auto amax = std::get<0>(blocks.abs().max(/*dim=*/1));
  auto scale = amax / 127.0 + eps;
  auto quantized =
      (blocks / scale.unsqueeze(1)).round().clamp(-128, 127).to(at::kChar);
  return std::make_tuple(quantized.reshape(input.sizes()), scale);
}

// Reverses quantize_per_block. out_dtype selects "float32" or "float64".
at::Tensor dequantize_per_block(
    const at::Tensor& qdata,
    const at::Tensor& scale,
    int block_size,
    const std::string& out_dtype) {
  TORCH_CHECK(out_dtype == "float32" || out_dtype == "float64",
              "unsupported out_dtype: ", out_dtype);
  auto dtype = out_dtype == "float32" ? at::kFloat : at::kDouble;
  auto blocks = qdata.to(dtype).reshape({-1, block_size});
  auto out = blocks * scale.unsqueeze(1).to(dtype);
  return out.reshape(qdata.sizes());
}

PYBIND11_MODULE(_C, m) {
  m.def("quantize_per_block", &quantize_per_block,
        "Blockwise int8 quantization returning (values, scales)",
        py::arg("input"), py::arg("block_size"), py::arg("eps"));
  m.def("dequantize_per_block", &dequantize_per_block,
        "Blockwise dequantization",
        py::arg("qdata"), py::arg("scale"), py::arg("block_size"),
        py::arg("out_dtype"));
}
