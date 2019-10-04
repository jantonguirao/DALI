// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATORS_CROP_SLICE_ATTR_H_
#define DALI_PIPELINE_OPERATORS_CROP_SLICE_ATTR_H_

#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/util/crop_window.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

class SliceAttr {
 protected:
  explicit SliceAttr(const OpSpec &spec);

  void ProcessArguments(MixedWorkspace &ws);
  void ProcessArguments(DeviceWorkspace &ws);
  void ProcessArguments(SampleWorkspace &ws);

  const CropWindowGenerator& GetCropWindowGenerator(size_t data_idx) const;

 private:
  void ProcessArgumentsHelper(int data_idx,
                              const float *slice_anchor_data,
                              const float *slice_shape_data);

  void VerifyArgsShape(const kernels::TensorShape<>& crop_anchor_shape,
                       const kernels::TensorShape<>& crop_shape_shape);

  size_t batch_size__;
  bool normalized_anchor_, normalized_shape_;
  std::vector<CropWindowGenerator> crop_window_generators_;
  std::vector<int> dims_;
  TensorLayout dim_names_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_SLICE_ATTR_H_
