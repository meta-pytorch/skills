// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace mixedext {

// C++ meta (fake) kernels: shape propagation only, no data access.
at::Tensor fused_scale_add_meta(
    const at::Tensor& x,
    const at::Tensor& y,
    double scale) {
  (void)scale;
  TORCH_CHECK(x.sizes() == y.sizes(), "fused_scale_add: shape mismatch");
  return at::empty_like(x);
}

TORCH_LIBRARY_IMPL(mixedext, Meta, m) {
  m.impl("fused_scale_add", &fused_scale_add_meta);
}

} // namespace mixedext
