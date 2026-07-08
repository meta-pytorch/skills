// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ops.h"

#include <ATen/ATen.h>

// SiLU(gate) * up — the gated activation used by SwiGLU MLP blocks.
at::Tensor silu_mul(const at::Tensor& gate, const at::Tensor& up) {
  TORCH_CHECK(gate.sizes() == up.sizes(),
              "gate and up must have the same shape");
  return at::silu(gate) * up;
}
