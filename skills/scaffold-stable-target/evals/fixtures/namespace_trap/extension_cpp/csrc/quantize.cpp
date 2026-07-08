// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <tuple>

namespace extension_cpp {

// torchao-style symmetric int8 quantization: returns (quantized, scales).
std::tuple<at::Tensor, at::Tensor> quantize_per_tensor_cpu(
    const at::Tensor& input,
    double scale) {
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(scale > 0.0, "scale must be positive");
  at::Tensor scaled = input / scale;
  at::Tensor quant = scaled.round().clamp(-128, 127).to(at::kChar);
  at::Tensor scales = at::full({}, scale, input.options());
  return std::make_tuple(quant, scales);
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("quantize_per_tensor", &quantize_per_tensor_cpu);
}

} // namespace extension_cpp
