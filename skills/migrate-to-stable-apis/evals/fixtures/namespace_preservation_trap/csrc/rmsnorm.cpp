// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// RMS normalization over the last dimension, CPU implementation.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

namespace fastkernels {

at::Tensor rmsnorm(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D [tokens, hidden]");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D [hidden]");
  TORCH_CHECK(input.size(1) == weight.size(0), "hidden size mismatch");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int64_t tokens = input.size(0);
  const int64_t hidden = input.size(1);

  at::Tensor output = at::empty_like(input);

  const float* in_ptr = input.data_ptr<float>();
  const float* w_ptr = weight.data_ptr<float>();
  float* out_ptr = output.data_ptr<float>();

  for (int64_t t = 0; t < tokens; ++t) {
    const float* row = in_ptr + t * hidden;
    float* out_row = out_ptr + t * hidden;
    float sum_sq = 0.0f;
    for (int64_t h = 0; h < hidden; ++h) {
      sum_sq += row[h] * row[h];
    }
    const float inv_rms =
        1.0f / std::sqrt(sum_sq / static_cast<float>(hidden) +
                         static_cast<float>(eps));
    for (int64_t h = 0; h < hidden; ++h) {
      out_row[h] = row[h] * inv_rms * w_ptr[h];
    }
  }

  return output;
}

}  // namespace fastkernels

TORCH_LIBRARY(fastkernels, m) {
  m.def("rmsnorm(Tensor input, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(fastkernels, CPU, m) {
  m.impl("rmsnorm", &fastkernels::rmsnorm);
}
