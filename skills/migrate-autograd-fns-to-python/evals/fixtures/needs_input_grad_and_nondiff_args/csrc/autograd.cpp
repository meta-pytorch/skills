// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>

// Defined in csrc/scalemix.cpp.
at::Tensor scalemix_cpu(const at::Tensor& a, const at::Tensor& b, int64_t shift);

class ScaleMixFunction : public torch::autograd::Function<ScaleMixFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& a,
      const at::Tensor& b,
      int64_t shift) {
    ctx->save_for_backward({a, b});
    at::AutoDispatchBelowADInplaceOrView guard;
    return scalemix_cpu(a, b, shift);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto a = saved[0];
    auto b = saved[1];
    auto grad = grad_outputs[0];
    at::Tensor grad_a, grad_b;
    if (ctx->needs_input_grad(0)) {
      // d(sin(a) + b^2 + shift)/da = cos(a)
      grad_a = grad * at::cos(a);
    }
    if (ctx->needs_input_grad(1)) {
      // d(sin(a) + b^2 + shift)/db = 2*b
      grad_b = grad * 2 * b;
    }
    // shift is a non-differentiable integer argument.
    return {grad_a, grad_b, at::Tensor()};
  }
};

at::Tensor scalemix_autograd(const at::Tensor& a, const at::Tensor& b, int64_t shift) {
  return ScaleMixFunction::apply(a, b, shift);
}

TORCH_LIBRARY_IMPL(gradext, Autograd, m) {
  m.impl("scalemix", &scalemix_autograd);
}
