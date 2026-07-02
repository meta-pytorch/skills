// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace notestext {

at::Tensor shift_relu(const at::Tensor& x, double shift) {
  TORCH_CHECK(x.scalar_type() == at::kFloat, "shift_relu: float32 only");
  auto out = at::empty_like(x);
  auto x_c = x.contiguous();
  const float* in = x_c.data_ptr<float>();
  float* dst = out.data_ptr<float>();
  float s = static_cast<float>(shift);
  for (int64_t i = 0; i < x_c.numel(); ++i) {
    float v = in[i] + s;
    dst[i] = v > 0.0f ? v : 0.0f;
  }
  return out;
}

TORCH_LIBRARY(notestext, m) {
  m.def("shift_relu(Tensor x, float shift) -> Tensor");
}

TORCH_LIBRARY_IMPL(notestext, CPU, m) {
  m.impl("shift_relu", &shift_relu);
}

} // namespace notestext
