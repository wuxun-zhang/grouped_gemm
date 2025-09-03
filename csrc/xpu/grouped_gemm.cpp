/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <c10/xpu/XPUStream.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <ATen/ATen.h>

// This is adapated from
// https://github.com/intel/cutlass-sycl/blob/sycl-develop/examples/sycl/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp.

#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"
#include "cutlass/kernel_hardware_info.h"

#include <cfloat>
#include "grouped_gemm.hpp"

namespace grouped_gemm {

// a -> input tensor with shape [num_experts * M, K]
// b -> weight tenor with shape [num_experts, K, N]
// c -> output tensor with shape [num_experts * M, N]
// batch_sizes -> a 1D tensor with shape [num_experts] which contains the
//   batchsize of each group trans_a -> whether to transpose the a tensor
// trans_b -> whether to transpose the b tensor
at::Tensor cutlass_grouped_gemm(const at::Tensor& a, const at::Tensor& b,
                                at::Tensor& c, const at::Tensor& batch_sizes,
                                bool trans_a, bool trans_b) {
  TORCH_CHECK(a.dim() == 2, "cutlass_grouped_gemm: a must be a 2D tensor");
  TORCH_CHECK(b.dim() == 3, "cutlass_grouped_gemm: b must be a 3D tensor");
  TORCH_CHECK(c.dim() == 2, "cutlass_grouped_gemm: c must be a 2D tensor");
  TORCH_CHECK(batch_sizes.dim() == 1,
              "cutlass_grouped_gemm: batch_sizes must be a 2D tensor");

  TORCH_CHECK(!trans_a, "cutlass_grouped_gemm: trans_a not supported");
  TORCH_CHECK(a.is_contiguous(), "cutlass_grouped_gemm: a must be contiguous");
  TORCH_CHECK(a.dtype() == torch::kBFloat16 && b.dtype() == torch::kBFloat16,
              "cutlass_grouped_gemm: only support bfloat16 for a/b");

  if (c.dtype() == torch::kBFloat16) {
    if (trans_b) {
      return cutlass_grouped_gemm_kernel<
          bfloat16_t, bfloat16_t, XE_8x16x16_F32BF16BF16F32_TT,
          XE_2D_U16x8x16_LD_N, XE_2D_U16x8x16_ST_N,
          cutlass::layout::ColumnMajor>(a, b, c, batch_sizes);
    } else {
      return cutlass_grouped_gemm_kernel<
          bfloat16_t, bfloat16_t, XE_8x16x16_F32BF16BF16F32_TT,
          XE_2D_U16x8x16_LD_N, XE_2D_U16x8x16_ST_N, cutlass::layout::RowMajor>(
          a, b, c, batch_sizes);
    }
  } else if (c.dtype() == torch::kFloat) {
    if (trans_b) {
      return cutlass_grouped_gemm_kernel<
          bfloat16_t, float, XE_8x16x16_F32BF16BF16F32_TT, XE_2D_U32x8x16_LD_N,
          XE_2D_U32x8x16_ST_N, cutlass::layout::ColumnMajor>(a, b, c,
                                                             batch_sizes);
    } else {
      return cutlass_grouped_gemm_kernel<
          bfloat16_t, float, XE_8x16x16_F32BF16BF16F32_TT, XE_2D_U32x8x16_LD_N,
          XE_2D_U32x8x16_ST_N, cutlass::layout::RowMajor>(a, b, c, batch_sizes);
    }
  } else {
    TORCH_CHECK(false, "cutlass_grouped_gemm: not supported dtype for c");
  }
  return c;
}

}  // namespace grouped_gemm
