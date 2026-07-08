// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>

#include <ATen/ATen.h>

// Applies rotary positional embedding to query and key IN PLACE.
// positions: [num_tokens] token positions used to index cos_sin_cache.
// query/key: [num_tokens, num_heads * head_size], overwritten in place.
// cos_sin_cache: [max_position, head_size] precomputed cos/sin values.
void rotary_embedding(
    const at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    const at::Tensor& cos_sin_cache,
    bool is_neox) {
  TORCH_CHECK(query.sizes() == key.sizes(),
              "query and key must have the same shape");
  auto cache = cos_sin_cache.index_select(0, positions);
  auto cos = cache.narrow(1, 0, head_size / 2).repeat_interleave(2, 1);
  auto sin = cache.narrow(1, head_size / 2, head_size / 2)
                 .repeat_interleave(2, 1);
  auto rotate = [&](at::Tensor& t) {
    auto shaped = t.view({t.size(0), -1, head_size});
    auto rotated = is_neox
        ? at::cat({-shaped.narrow(2, head_size / 2, head_size / 2),
                   shaped.narrow(2, 0, head_size / 2)},
                  2)
        : shaped.roll(1, 2);
    shaped.mul_(cos.unsqueeze(1)).add_(rotated * sin.unsqueeze(1));
  };
  rotate(query);
  rotate(key);
}

PYBIND11_MODULE(_C, m) {
  m.def("rotary_embedding", &rotary_embedding,
        "Apply rotary positional embedding to query and key in place",
        py::arg("positions"), py::arg("query"), py::arg("key"),
        py::arg("head_size"), py::arg("cos_sin_cache"), py::arg("is_neox"));
}
