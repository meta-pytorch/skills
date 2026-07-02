// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Out-variant ReLU over float32 tensors.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>

namespace tinyops {

void relu_out(const at::Tensor& input, at::Tensor& out) {
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(input.numel() == out.numel(), "input/out numel mismatch");

  const float* in_ptr = input.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();

  const int64_t n = input.numel();
  for (int64_t i = 0; i < n; ++i) {
    out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
  }
}

}  // namespace tinyops

TORCH_LIBRARY_IMPL(tinyops, CPU, m) {
  m.impl("relu_out", &tinyops::relu_out);
}
