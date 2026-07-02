// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>

// Defined in csrc/tanhmul.cpp.
at::Tensor mytanhmul_cpu(const at::Tensor& a, const at::Tensor& b);

class MyTanhMulFunction : public torch::autograd::Function<MyTanhMulFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& a,
      const at::Tensor& b) {
    at::Tensor output;
    {
      at::AutoDispatchBelowADInplaceOrView guard;
      output = mytanhmul_cpu(a, b);
    }
    // Save the output too: the tanh gradient reuses it instead of
    // recomputing tanh(a * b) in backward.
    ctx->save_for_backward({a, b, output});
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto a = saved[0];
    auto b = saved[1];
    auto output = saved[2];
    auto grad = grad_outputs[0];
    // d tanh(u)/du = 1 - tanh(u)^2 = 1 - output^2, with u = a * b.
    auto grad_u = grad * (1 - output * output);
    return {grad_u * b, grad_u * a};
  }
};

at::Tensor mytanhmul_autograd(const at::Tensor& a, const at::Tensor& b) {
  return MyTanhMulFunction::apply(a, b);
}

TORCH_LIBRARY_IMPL(gradext, Autograd, m) {
  m.impl("mytanhmul", &mytanhmul_autograd);
}
