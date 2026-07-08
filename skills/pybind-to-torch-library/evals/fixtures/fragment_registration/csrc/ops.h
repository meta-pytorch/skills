// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/ATen.h>

at::Tensor rms_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps);

at::Tensor silu_mul(const at::Tensor& gate, const at::Tensor& up);
