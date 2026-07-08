// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace windowops {

at::Tensor window_avg_meta(const at::Tensor& x, int64_t kernel, int64_t stride, int64_t pad) {
  TORCH_CHECK(x.dim() == 2, "window_avg: expected a 2-D input");
  TORCH_CHECK(kernel > 0 && stride > 0 && pad >= 0, "window_avg: bad window parameters");
  auto B = x.sym_sizes()[0];
  auto L = x.sym_sizes()[1];
  auto out_len = (L + 2 * pad - kernel) / stride + 1;
  return at::empty_symint({B, out_len}, x.options());
}

at::Tensor pack_pair_meta(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 2, "pack_pair: expected 2-D inputs");
  TORCH_CHECK(a.sym_sizes() == b.sym_sizes(), "pack_pair: a and b must have the same shape");
  auto M = a.sym_sizes()[0];
  auto N = a.sym_sizes()[1];
  return at::empty_symint({M, N * 2 + 1}, a.options());
}

TORCH_LIBRARY_IMPL(windowops, Meta, m) {
  m.impl("window_avg", &window_avg_meta);
  m.impl("pack_pair", &pack_pair_meta);
}

} // namespace windowops
