// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>

// Defined in csrc/muladd.cpp.
at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c);

class MyMulAddFunction : public torch::autograd::Function<MyMulAddFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& a,
      const at::Tensor& b,
      double c) {
    ctx->save_for_backward({a, b});
    at::AutoDispatchBelowADInplaceOrView guard;
    return mymuladd_cpu(a, b, c);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto a = saved[0];
    auto b = saved[1];
    auto grad = grad_outputs[0];
    // d(a*b + c)/da = b, d(a*b + c)/db = a, c is a non-differentiable scalar.
    return {grad * b, grad * a, at::Tensor()};
  }
};

at::Tensor mymuladd_autograd(const at::Tensor& a, const at::Tensor& b, double c) {
  return MyMulAddFunction::apply(a, b, c);
}

TORCH_LIBRARY_IMPL(gradext, Autograd, m) {
  m.impl("mymuladd", &mymuladd_autograd);
}
