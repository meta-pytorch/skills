// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// FlashAttention-style fused attention forward, CPU reference implementation.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <optional>

namespace fastattn {

at::Tensor attention_fwd(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    std::optional<at::Tensor> out,
    double softmax_scale) {
  TORCH_CHECK(query.dim() == 4, "query must be 4D [batch, heads, seq, dim]");
  TORCH_CHECK(key.dim() == 4, "key must be 4D [batch, heads, seq, dim]");
  TORCH_CHECK(value.dim() == 4, "value must be 4D [batch, heads, seq, dim]");
  TORCH_CHECK(query.scalar_type() == at::kFloat, "query must be float32");
  TORCH_CHECK(key.scalar_type() == at::kFloat, "key must be float32");
  TORCH_CHECK(value.scalar_type() == at::kFloat, "value must be float32");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");

  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t seq_q = query.size(2);
  const int64_t head_dim = query.size(3);
  const int64_t seq_k = key.size(2);

  TORCH_CHECK(key.size(3) == head_dim, "key head dim mismatch");
  TORCH_CHECK(value.size(2) == seq_k, "value seq length mismatch");
  TORCH_CHECK(value.size(3) == head_dim, "value head dim mismatch");

  at::Tensor output;
  if (out.has_value()) {
    TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
    TORCH_CHECK(
        out->numel() == batch * heads * seq_q * head_dim,
        "out has wrong number of elements");
    output = out.value();
  } else {
    output = at::empty({batch, heads, seq_q, head_dim}, query.options());
  }

  at::Tensor scores = at::empty({batch, heads, seq_q, seq_k}, query.options());
  at::Tensor probs = at::empty_like(scores);

  const float* q_ptr = query.data_ptr<float>();
  const float* k_ptr = key.data_ptr<float>();
  const float* v_ptr = value.data_ptr<float>();
  float* s_ptr = scores.data_ptr<float>();
  float* p_ptr = probs.data_ptr<float>();
  float* o_ptr = output.data_ptr<float>();

  const float scale = static_cast<float>(softmax_scale);

  for (int64_t bh = 0; bh < batch * heads; ++bh) {
    const float* q_bh = q_ptr + bh * seq_q * head_dim;
    const float* k_bh = k_ptr + bh * seq_k * head_dim;
    const float* v_bh = v_ptr + bh * seq_k * head_dim;
    float* s_bh = s_ptr + bh * seq_q * seq_k;
    float* p_bh = p_ptr + bh * seq_q * seq_k;
    float* o_bh = o_ptr + bh * seq_q * head_dim;

    // scores = scale * (Q @ K^T)
    for (int64_t i = 0; i < seq_q; ++i) {
      for (int64_t j = 0; j < seq_k; ++j) {
        float acc = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
          acc += q_bh[i * head_dim + d] * k_bh[j * head_dim + d];
        }
        s_bh[i * seq_k + j] = acc * scale;
      }
    }

    // probs = softmax(scores, dim=-1); output = probs @ V
    for (int64_t i = 0; i < seq_q; ++i) {
      float row_max = -INFINITY;
      for (int64_t j = 0; j < seq_k; ++j) {
        row_max = std::max(row_max, s_bh[i * seq_k + j]);
      }
      float denom = 0.0f;
      for (int64_t j = 0; j < seq_k; ++j) {
        const float e = std::exp(s_bh[i * seq_k + j] - row_max);
        p_bh[i * seq_k + j] = e;
        denom += e;
      }
      for (int64_t d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int64_t j = 0; j < seq_k; ++j) {
          acc += (p_bh[i * seq_k + j] / denom) * v_bh[j * head_dim + d];
        }
        o_bh[i * head_dim + d] = acc;
      }
    }
  }

  return output;
}

}  // namespace fastattn

TORCH_LIBRARY(fastattn, m) {
  m.def(
      "attention_fwd(Tensor query, Tensor key, Tensor value, Tensor? out, "
      "float softmax_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(fastattn, CPU, m) {
  m.impl("attention_fwd", &fastattn::attention_fwd);
}
