// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status ReduceSumOnLastAxis(const int8_t* a, int32_t* a_row_sum, const int8_t* offset, const MatMulComputeHelper& helper);
Status ReduceSumOnSecondToLastAxis(const int8_t* a, int32_t* b_col_sum, const int8_t* offset, const MatMulComputeHelper& helper);
Status InitializeOutput(const int32_t* row_sum,
                        const int32_t* col_sum,
                        int32_t* output,
                        const int8_t* a_offset,
                        const int8_t* b_offset,
                        const MatMulComputeHelper& helper);

Status PadMatrixInLeadingDimension(const int8_t* src, int8_t* dst, int row, int col, int pad_size);

}  // namespace cuda
}  // namespace onnxruntime
