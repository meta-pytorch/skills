// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

// Defined in csrc/mymul.cpp.
at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b);

// Route through the dispatcher so the registered
// gradext::mymul_backward_kernel kernel is used (it may have CPU/CUDA
// variants registered).
static at::Tensor call_mymul_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("gradext::mymul_backward_kernel", "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>();
  return op.call(grad, other);
}

class MyMulFunction : public torch::autograd::Function<MyMulFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& a,
      const at::Tensor& b) {
    ctx->save_for_backward({a, b});
    at::AutoDispatchBelowADInplaceOrView guard;
    return mymul_cpu(a, b);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto a = saved[0];
    auto b = saved[1];
    auto grad = grad_outputs[0];
    // Gradients are computed by the extension's fused backward kernel,
    // NOT by open-coded tensor math.
    auto grad_a = call_mymul_backward_kernel(grad, b);
    auto grad_b = call_mymul_backward_kernel(grad, a);
    return {grad_a, grad_b};
  }
};

at::Tensor mymul_autograd(const at::Tensor& a, const at::Tensor& b) {
  return MyMulFunction::apply(a, b);
}

TORCH_LIBRARY_IMPL(gradext, Autograd, m) {
  m.impl("mymul", &mymul_autograd);
}
