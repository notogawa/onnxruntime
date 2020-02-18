// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer.h"
#include "matmul_integer.cuh"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<int8_t, int8_t>);

template <>
Status MatMulInteger<int8_t, int8_t>::ComputeInternal(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // validate zero points
  const int8_t* a_offset = nullptr;
  const int8_t* b_offset = nullptr;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = a_zero_point->template Data<int8_t>();
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    b_offset = b_zero_point->template Data<int8_t>();
  }

  const int8_t* a_ptr = a->template Data<int8_t>();
  const int8_t* b_ptr = b->template Data<int8_t>();
  int32_t* output_ptr = Y->template MutableData<int32_t>();

  // intialize output c[i,j] to
  // k*a_offset*b_offset -
  // b_offset * (a[i,0] + a[i,1] ...+a[i,k]) -
  // a_offset * (b[0,j] + b[1,j] ... + b[k,j])
  IAllocatorUniquePtr<int32_t> a_row_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.N());
  IAllocatorUniquePtr<int32_t> b_col_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.M());
  ReduceSumOnLastAxis(a_ptr, a_row_buf.get(), b_offset, helper);
  ReduceSumOnSecondToLastAxis(b_ptr, b_col_buf.get(), a_offset, helper);

  InitializeOutput(a_row_buf.get(),
                   b_col_buf.get(),
                   output_ptr,
                   a_offset,
                   b_offset,
                   helper);

  // pad A and B to make their leading dimension be multiples of 32
  // because cublasGemmEx requires:
  // 1. leading dimension is multiples of 4
  // 2. A, B is 32-bit aligned
  const int64_t align_size = 32;
  int64_t a_pad_size = 0;
  int64_t b_pad_size = 0;
  int64_t a_pad_per_matrix = 0;
  int64_t b_pad_per_matrix = 0;
  IAllocatorUniquePtr<int8_t> a_padded;
  IAllocatorUniquePtr<int8_t> b_padded;

  int64_t pad_size = align_size - helper.K() % align_size;
  if (pad_size != align_size) {
    int64_t row = a->Shape().Size() / helper.K();
    a_padded = GetScratchBuffer<int8_t>(a->Shape().Size() + row * pad_size);
    PadMatrixInLeadingDimension(a_ptr, a_padded.get(), static_cast<int>(row), static_cast<int>(helper.K()), static_cast<int>(pad_size));
    a_ptr = a_padded.get();
    a_pad_size = pad_size;
    a_pad_per_matrix = pad_size * helper.M();
  }

  pad_size = align_size - helper.N() % align_size;
  if (pad_size != align_size) {
    int64_t row = b->Shape().Size() / helper.N();
    b_padded = GetScratchBuffer<int8_t>(b->Shape().Size() + row * pad_size);
    PadMatrixInLeadingDimension(b_ptr, b_padded.get(), static_cast<int>(row), static_cast<int>(helper.N()), static_cast<int>(pad_size));
    b_ptr = b_padded.get();
    b_pad_size = pad_size;
    b_pad_per_matrix = pad_size * helper.K();
  }

  int alpha = 1;
  int beta = 1;

  for (int batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
        Base::CublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &alpha,
        b_ptr + helper.RightOffsets()[batch] + b_pad_per_matrix * batch,
        CUDA_R_8I,
        static_cast<int>(helper.N() + b_pad_size),
        a_ptr + helper.LeftOffsets()[batch] + a_pad_per_matrix * batch,
        CUDA_R_8I,
        static_cast<int>(helper.K() + a_pad_size),
        &beta,
        output_ptr + helper.OutputOffsets()[batch],
        CUDA_R_32I,
        static_cast<int>(helper.N()),
        CUDA_R_32I,
        CUBLAS_GEMM_DFALT));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
