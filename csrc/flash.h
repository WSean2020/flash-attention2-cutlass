#pragma once

#include <cstdint>

struct Flash_fwd_params {
  bool is_bf16;
  using index_t = uint32_t;

  size_t bs;
  size_t head;
  size_t q_seqlen;
  size_t dim;

  size_t k_head;
  size_t k_seqlen;

  // TODO: review the impl of flash
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
  // different from nheads (query).
  size_t h_h_k_ratio; // precompute head / k_head,
  size_t flat_seqlen;
  size_t kv_head_stride;
  size_t qo_head_stride;

  size_t bs_stride;
  size_t head_stride;
  size_t seqlen_stride;
  size_t dim_stride;

  float softmax_scale;
  float softmax_scale_log2;

  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ v_ptr;

  void *__restrict__ out_ptr;
  void *__restrict__ softmax_lse_ptr;
  void *__restrict__ score_max;
  void *__restrict__ score_sum;

  bool is_causal;
};
