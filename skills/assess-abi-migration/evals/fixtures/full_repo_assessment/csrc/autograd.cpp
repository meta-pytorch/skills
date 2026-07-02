// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace mixedext {

// C++ autograd wiring for fused_scale_add. The backward of
// out = x * scale + y is grad_x = grad * scale, grad_y = grad.
class FusedScaleAddFunction
    : public torch::autograd::Function<FusedScaleAddFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& y,
      double scale) {
    ctx->saved_data["scale"] = scale;
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("mixedext::fused_scale_add", "")
                         .typed<at::Tensor(const at::Tensor&, const at::Tensor&, double)>();
    return op.call(x, y, scale);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    double scale = ctx->saved_data["scale"].toDouble();
    auto grad = grad_outputs[0];
    return {grad * scale, grad, at::Tensor()};
  }
};

at::Tensor fused_scale_add_autograd(
    const at::Tensor& x,
    const at::Tensor& y,
    double scale) {
  return FusedScaleAddFunction::apply(x, y, scale);
}

TORCH_LIBRARY_IMPL(mixedext, Autograd, m) {
  m.impl("fused_scale_add", &fused_scale_add_autograd);
}

} // namespace mixedext
