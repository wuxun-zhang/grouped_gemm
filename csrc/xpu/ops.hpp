#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>

namespace grouped_gemm {
at::Tensor cutlass_grouped_gemm(const at::Tensor& a, const at::Tensor& b,
                                at::Tensor& c, const at::Tensor& batch_sizes,
                                bool trans_a = false, bool trans_b = false);
}  // namespace grouped_gemm
