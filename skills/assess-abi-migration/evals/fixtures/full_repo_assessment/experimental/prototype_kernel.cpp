// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Experimental scratchpad — NOT listed in setup.py sources and never shipped.
#include <ATen/ATen.h>

namespace mixedext_experimental {

at::Tensor prototype_double(const at::Tensor& x) {
  TORCH_CHECK(x.defined(), "prototype_double: undefined tensor");
  return x + x;
}

} // namespace mixedext_experimental
