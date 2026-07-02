// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

namespace gapext {

// (a) at::empty_like — has a same-named stable counterpart in
// torch::stable::ops.h, so it must NOT be reported as a coverage gap.
at::Tensor clamp_shift(const at::Tensor& x, double lo, double hi) {
  TORCH_CHECK(x.scalar_type() == at::kFloat, "clamp_shift: float32 only");
  auto out = at::empty_like(x);

  // (b) at::TensorIterator — libtorch-internal iteration machinery with no
  // stable equivalent (structural gap).
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_input(x)
                  .build();
  at::native::cpu_kernel(iter, [lo, hi](float v) -> float {
    float c = v < lo ? static_cast<float>(lo)
                     : (v > hi ? static_cast<float>(hi) : v);
    return c + 1.0f;
  });
  return out;
}

// (c) at::empty_strided — no stable same-named factory; the documented
// workaround is new_empty with transposed sizes followed by transpose.
at::Tensor make_colmajor_buffer(const at::Tensor& ref) {
  TORCH_CHECK(ref.dim() == 2, "make_colmajor_buffer: expected 2-D ref");
  int64_t rows = ref.size(0);
  int64_t cols = ref.size(1);
  auto out = at::empty_strided({rows, cols}, {1, rows}, ref.options());
  return out;
}

TORCH_LIBRARY(gapext, m) {
  m.def("clamp_shift(Tensor x, float lo, float hi) -> Tensor");
  m.def("make_colmajor_buffer(Tensor ref) -> Tensor");
}

TORCH_LIBRARY_IMPL(gapext, CPU, m) {
  m.impl("clamp_shift", &clamp_shift);
  m.impl("make_colmajor_buffer", &make_colmajor_buffer);
}

} // namespace gapext
