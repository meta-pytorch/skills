// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ops.h"

#include <ATen/ATen.h>

// Root-mean-square layer norm over the last dimension.
at::Tensor rms_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps) {
  TORCH_CHECK(input.size(-1) == weight.size(0),
              "weight must match the last dimension of input");
  auto variance = input.pow(2).mean(-1, /*keepdim=*/true);
  return input * at::rsqrt(variance + eps) * weight;
}
