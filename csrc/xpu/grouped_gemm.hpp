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

// This is adapated from
// https://github.com/intel/cutlass-sycl/blob/sycl-develop/examples/sycl/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp.
#include <c10/xpu/XPUStream.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <ATen/ATen.h>

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

namespace grouped_gemm {

using namespace cute;

template <typename ElementA, typename ElementB, typename ElementC,
          typename StrideA, typename StrideB, typename StrideC,
          typename ProblemShape>
struct grouped_gemm_tensors {
  torch::Tensor ptr_a, ptr_b, ptr_c;
  torch::Tensor stride_a, stride_b, stride_c;
  torch::Tensor problem_sizes;

  grouped_gemm_tensors(int num_experts, const torch::Device& device) {
    auto options = torch::TensorOptions().dtype(torch::kChar).device(device);
    auto bytes = num_experts * sizeof_bits<ElementA>::value / 8;
    ptr_a = torch::empty(bytes, options);
    bytes = num_experts * sizeof_bits<ElementB>::value / 8;
    ptr_b = torch::empty(bytes, options);
    bytes = num_experts * sizeof_bits<ElementC>::value / 8;
    ptr_c = torch::empty(bytes, options);
    bytes = num_experts * sizeof_bits<StrideA>::value / 8;
    stride_a = torch::empty(bytes, options);
    bytes = num_experts * sizeof_bits<StrideB>::value / 8;
    stride_b = torch::empty(bytes, options);
    bytes = num_experts * sizeof_bits<StrideC>::value / 8;
    stride_c = torch::empty(bytes, options);
    bytes = num_experts * sizeof_bits<ProblemShape>::value / 8;
    problem_sizes = torch::empty(bytes, options);
  }
};

using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group

std::vector<typename ProblemShape::UnderlyingProblemShape> MakeProblemSizes(
    const torch::Tensor& b, const torch::Tensor& batch_sizes) {
  const size_t num_experts = batch_sizes.size(0);
  const int k = b.size(1);
  const int n = b.size(2);
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes(
      num_experts);
  for (int i = 0; i < num_experts; ++i) {
    problem_sizes[i] = {(int)batch_sizes.data_ptr<int64_t>()[i], n, k};
  }
  return problem_sizes;
}

template <typename T>
void CopyToDevice(const std::vector<T>& x, const torch::Tensor& device_tensor) {
  TORCH_CHECK(device_tensor.dtype() == torch::kChar,
              "device_tensor must be int8 dtype");
  auto bytes = x.size() * sizeof_bits<T>::value / 8;
  syclcompat::memcpy_async(device_tensor.data_ptr(), x.data(), bytes,
                           c10::xpu::getCurrentXPUStream().queue());
}

template <typename Gemm, typename GroupedGemmTensorType>
typename Gemm::Arguments MakeArguments(
    const GroupedGemmTensorType& grouped_gemm_tensors, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& c,
    const torch::Tensor& batch_sizes,
    const std::vector<typename ProblemShape::UnderlyingProblemShape>&
        problem_sizes_host) {
  int64_t num_experts = problem_sizes_host.size();

  // Create the host arrays of pointer data.
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  std::vector<int64_t> offsets_a(num_experts);
  std::vector<int64_t> offsets_b(num_experts);
  std::vector<int64_t> offsets_c(num_experts);
  std::vector<StrideA> stride_a_host(num_experts);
  std::vector<StrideB> stride_b_host(num_experts);
  std::vector<StrideC> stride_c_host(num_experts);

  int64_t elements_a = 0, elements_b = 0, elements_c = 0;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC =
      typename Gemm::CollectiveEpilogue::ElementOutput;  // assume C and D have
                                                         // the same type
  std::vector<ElementA*> ptr_a_host(num_experts);
  std::vector<ElementB*> ptr_b_host(num_experts);
  std::vector<ElementC*> ptr_c_host(num_experts);

  for (int i = 0; i < num_experts; ++i) {
    auto problem = problem_sizes_host[i];
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    stride_a_host[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_b_host[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_c_host[i] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});

    offsets_a[i] = elements_a;
    offsets_b[i] = elements_b;
    offsets_c[i] = elements_c;

    ptr_a_host[i] = (ElementA*)a.data_ptr() + offsets_a[i];
    ptr_b_host[i] = (ElementB*)b.data_ptr() + offsets_b[i];
    ptr_c_host[i] = (ElementC*)c.data_ptr() + offsets_c[i];

    elements_a += M * K;
    elements_b += K * N;
    elements_c += M * N;
  }

  CopyToDevice(stride_a_host, grouped_gemm_tensors.stride_a);
  CopyToDevice(stride_b_host, grouped_gemm_tensors.stride_b);
  CopyToDevice(stride_c_host, grouped_gemm_tensors.stride_c);
  CopyToDevice(ptr_a_host, grouped_gemm_tensors.ptr_a);
  CopyToDevice(ptr_b_host, grouped_gemm_tensors.ptr_b);
  CopyToDevice(ptr_c_host, grouped_gemm_tensors.ptr_c);
  CopyToDevice(problem_sizes_host, grouped_gemm_tensors.problem_sizes);

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;

  // same epilogue arguments for all groups
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  // Single alpha and beta for all groups
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  using RasterOrderOptions =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<
          ProblemShape>::RasterOrderOptions;

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {(int)num_experts,
       (typename ProblemShape::UnderlyingProblemShape*)
           grouped_gemm_tensors.problem_sizes.data_ptr(),
       problem_sizes_host.data()},
      {(const ElementA**)grouped_gemm_tensors.ptr_a.data_ptr(),
       (StrideA*)grouped_gemm_tensors.stride_a.data_ptr(),
       (const ElementB**)grouped_gemm_tensors.ptr_b.data_ptr(),
       (StrideB*)grouped_gemm_tensors.stride_b.data_ptr()},
      {fusion_args, (const ElementC**)grouped_gemm_tensors.ptr_c.data_ptr(),
       (StrideC*)grouped_gemm_tensors.stride_c.data_ptr(),
       (ElementC**)grouped_gemm_tensors.ptr_c.data_ptr(),
       (StrideC*)grouped_gemm_tensors.stride_c.data_ptr()},
      hw_info,
      {1, RasterOrderOptions::AlongN}};

  return arguments;
}

template <typename DataType, typename OutputType, typename Xe_MMA_Atom,
          typename EpilogueLDType, typename EpilogueSTDtype,
          typename LayoutTypeB>
at::Tensor cutlass_grouped_gemm_kernel(const at::Tensor& a, const at::Tensor& b,
                                       const at::Tensor& c,
                                       const at::Tensor& batch_sizes) {
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                               // group

  using ElementAccumulator = float;      // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementA = DataType;  // <- data type of elements in input matrix A
  using ElementB = DataType;  // <- data type of elements in input matrix B
  using ElementOutput =
      OutputType;  // <- data type of elements in output matrix D

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = LayoutTypeB;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB =
      conditional_t<std::is_same_v<LayoutB, cutlass::layout::ColumnMajor>,
                    XE_2D_U16x16x16_LD_T, XE_2D_U16x16x16_LD_V>;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      TiledMMA<MMA_Atom<Xe_MMA_Atom>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
               Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>,
                    Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;

  constexpr int PipelineStages = 2;
  // Dispatch to grouped gemm algorithm
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16Group<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementOutput, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape,
      decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy, TileShape, ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC*>, ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD*>, FusionCallBacks, EpilogueLDType,
      void, void, EpilogueSTDtype, void, void>;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape, ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA*>, ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB*>, TiledMma, GmemTiledCopyA, void,
      void, cute::identity,                       // A
      GmemTiledCopyB, void, void, cute::identity  // B
      >;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using GroupedGemmTensorType =
      grouped_gemm_tensors<ElementA, ElementB, ElementOutput,
                           typename Gemm::GemmKernel::InternalStrideA,
                           typename Gemm::GemmKernel::InternalStrideB,
                           typename Gemm::GemmKernel::InternalStrideC,
                           typename ProblemShape::UnderlyingProblemShape>;

  GroupedGemmTensorType grouped_gemm_tensors{(int)batch_sizes.size(0),
                                             a.device()};

  std::vector<typename ProblemShape::UnderlyingProblemShape>
      problem_sizes_host = MakeProblemSizes(b, batch_sizes);
  auto arguments = MakeArguments<Gemm, GroupedGemmTensorType>(
      grouped_gemm_tensors, a, b, c, batch_sizes, problem_sizes_host);
  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
    TORCH_CHECK(
        false,
        "Cannot implement Cutlass grouped GEMM with the given arguments");
  }

  if (gemm_op.initialize(arguments, workspace.get()) !=
      cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to initialize Cutlass grouped GEMM");
  }

  if (gemm_op.run(&c10::xpu::getCurrentXPUStream().queue()) !=
      cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to run Cutlass grouped GEMM");
  }
  return c;
}

}  // namespace grouped_gemm
