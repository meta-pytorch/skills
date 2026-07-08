// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace windowops {

at::Tensor window_avg_cpu(const at::Tensor& x, int64_t kernel, int64_t stride, int64_t pad) {
  TORCH_CHECK(x.dim() == 2, "window_avg: expected a 2-D input");
  TORCH_CHECK(kernel > 0 && stride > 0 && pad >= 0, "window_avg: bad window parameters");
  auto padded = at::constant_pad_nd(x, {pad, pad}, 0);
  return padded.unfold(-1, kernel, stride).mean(-1);
}

at::Tensor pack_pair_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 2, "pack_pair: expected 2-D inputs");
  TORCH_CHECK(a.sizes() == b.sizes(), "pack_pair: a and b must have the same shape");
  auto sep = at::ones({a.size(0), 1}, a.options());
  return at::cat({a, sep, b}, 1);
}

TORCH_LIBRARY(windowops, m) {
  m.def("window_avg(Tensor x, int kernel, int stride, int pad) -> Tensor");
  m.def("pack_pair(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(windowops, CPU, m) {
  m.impl("window_avg", &window_avg_cpu);
  m.impl("pack_pair", &pack_pair_cpu);
}

} // namespace windowops
