// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Produces a column-major scaled copy of a 2D tensor, suitable for handoff
// to BLAS-style column-major consumers.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>

namespace colmajor {

at::Tensor scale_to_colmajor(const at::Tensor& input, double scale) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);

  // Column-major output: shape {rows, cols} with strides {1, rows}.
  at::Tensor output =
      at::empty_strided({rows, cols}, {1, rows}, input.options());

  const float* in_ptr = input.data_ptr<float>();
  float* out_ptr = output.data_ptr<float>();
  const float s = static_cast<float>(scale);

  const int64_t out_stride_0 = output.strides()[0];
  const int64_t out_stride_1 = output.strides()[1];

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out_ptr[i * out_stride_0 + j * out_stride_1] =
          in_ptr[i * cols + j] * s;
    }
  }

  return output;
}

}  // namespace colmajor

TORCH_LIBRARY(colmajor, m) {
  m.def("scale_to_colmajor(Tensor input, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(colmajor, CPU, m) {
  m.impl("scale_to_colmajor", &colmajor::scale_to_colmajor);
}
