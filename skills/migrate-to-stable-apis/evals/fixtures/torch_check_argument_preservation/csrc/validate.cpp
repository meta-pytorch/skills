// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Argument validation helpers with detailed, multi-part error messages.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>

namespace guardrails {

at::Tensor checked_flatten(const at::Tensor& input, int64_t expected_rank) {
  TORCH_CHECK(
      input.dim() == expected_rank,
      "checked_flatten: expected rank ",
      expected_rank,
      ", but got tensor of rank ",
      input.dim(),
      " (shape has ",
      input.numel(),
      " total elements)");
  TORCH_CHECK(
      input.is_contiguous(),
      "checked_flatten: input must be contiguous; call .contiguous() first, "
      "then retry");
  TORCH_CHECK(
      input.numel() > 0 && input.numel() < (int64_t{1} << 31),
      "checked_flatten: numel must be in (0, 2^31), got ",
      input.numel(),
      "; empty and oversized tensors are not supported");
  TORCH_CHECK(
      input.size(0) % 2 == 0 || input.size(0) == 1,
      "checked_flatten: leading dim must be even or 1, got ",
      input.size(0));

  const int64_t n = input.numel();
  at::Tensor out = at::empty_like(input);
  const float* in_ptr = input.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    out_ptr[i] = in_ptr[i];
  }
  return out;
}

}  // namespace guardrails

TORCH_LIBRARY(guardrails, m) {
  m.def("checked_flatten(Tensor input, int expected_rank) -> Tensor");
}

TORCH_LIBRARY_IMPL(guardrails, CPU, m) {
  m.impl("checked_flatten", &guardrails::checked_flatten);
}
