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

#include <vector>
#include <utility>
#include <sstream>
#include "dali/core/format.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operators/crop/slice_attr.h"

namespace dali {

/*
DALI_SCHEMA(SliceAttr)
.DocStr(R"code(Slice attributes placeholder)code")
.AddOptionalArg("dims",
    R"code(Order of dimensions used for anchor and shape slice inputs, as dimension indexes)code",
    std::vector<int>{1, 0})
.AddOptionalArg("dim_names",
    R"code(Order of dimensions used for anchor and shape slice inputs, as described in layout.
If provided, `dim_names` takes higher priority than `dims`)code",
    "WH")
.AddOptionalArg("normalized_anchor",
    R"code(Whether or not the `anchor` input should be interpreted as normalized (range [0.0, 1.0])
or absolute coordinates)code",
    true)
.AddOptionalArg("normalized_shape",
    R"code(Whether or not the `shape` input should be interpreted as normalized (range [0.0, 1.0])
or absolute coordinates)code",
    true);
*/

SliceAttr::SliceAttr(const OpSpec &spec)
    : batch_size__(spec.GetArgument<int>("batch_size"))
    , normalized_anchor_(spec.GetArgument<bool>("normalized_anchor"))
    , normalized_shape_(spec.GetArgument<bool>("normalized_shape"))
    , crop_window_generators_(batch_size__) {
  const bool has_dims_arg = spec.HasArgument("dims");
  const bool has_dim_names_arg = spec.HasArgument("dim_names");
  // Process `dim_names` if provided, or if neither `dir_names` nor `dims` are
  if (has_dim_names_arg || !has_dims_arg) {
    dim_names_ = spec.GetArgument<TensorLayout>("dim_names");
    dims_ = {};
  } else {
    // Process `dims` only if provided and `dim_names` isn't
    dims_ = spec.GetRepeatedArgument<int>("dims");
    dim_names_ = TensorLayout{};
  }
}
/*

void SliceAttr::ProcessArguments(MixedWorkspace &ws) {
  DALI_ENFORCE(ws.NumInput() == 3,
    "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
  for (std::size_t data_idx = 0; data_idx < batch_size__; data_idx++) {
    const auto &images = ws.Input<CPUBackend>(0, data_idx);
    const auto &crop_anchor = ws.Input<CPUBackend>(1, data_idx);
    const auto &crop_shape = ws.Input<CPUBackend>(2, data_idx);
    VerifyArgsShape(crop_anchor.shape(), crop_shape.shape());
    ProcessArgumentsHelper(data_idx, crop_anchor.data<float>(), crop_shape.data<float>());
  }
}

void SliceAttr::ProcessArguments(DeviceWorkspace &ws) {
  DALI_ENFORCE(ws.NumInput() == 3,
    "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
  const auto &images = ws.Input<GPUBackend>(0);
  const auto &crop_anchor = ws.Input<CPUBackend>(1);
  const auto &crop_shape = ws.Input<CPUBackend>(2);
  for (std::size_t data_idx = 0; data_idx < batch_size__; data_idx++) {
    VerifyArgsShape(crop_anchor.tensor_shape(data_idx), crop_shape.tensor_shape(data_idx));
    ProcessArgumentsHelper(data_idx, crop_anchor.tensor<float>(data_idx),
                           crop_shape.tensor<float>(data_idx));
  }
}

void SliceAttr::ProcessArguments(SampleWorkspace &ws) {
  DALI_ENFORCE(ws.NumInput() == 3,
    "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
  const auto &images = ws.Input<CPUBackend>(0);
  const auto &crop_anchor = ws.Input<CPUBackend>(1);
  const auto &crop_shape = ws.Input<CPUBackend>(2);
  VerifyArgsShape(crop_anchor.shape(), crop_shape.shape());
  ProcessArgumentsHelper(ws.data_idx(), crop_anchor.data<float>(), crop_shape.data<float>());
}

const CropWindowGenerator& SliceAttr::GetCropWindowGenerator(size_t data_idx) const {
  DALI_ENFORCE(data_idx < crop_window_generators_.size());
  return crop_window_generators_[data_idx];
}

void SliceAttr::ProcessArgumentsHelper(int data_idx,
                                       const float *slice_anchor_data,
                                       const float *slice_shape_data) {
  crop_window_generators_[data_idx] =
    [this, slice_anchor_data, slice_shape_data](const kernels::TensorShape<> &shape,
                                                const TensorLayout& shape_layout) {
      CropWindow slice;
      slice.anchor = std::vector<int64_t>(shape.size(), 0);
      slice.shape = shape;
      if (!dim_names_.empty()) {
        dims_ = {};
        for (auto dim_name : dim_names_) {
          auto dim_idx = shape_layout.find(dim_name);
          DALI_ENFORCE(dim_idx >= 0,
            make_string("Requested to slice dimension", dim_name,
              "which is not present in the shape layout", shape_layout.c_str()));
          dims_.push_back(dim_idx);
        }
      }

      for (size_t i = 0; i < dims_.size(); i++) {
        auto dim = dims_[i];
        float anchor_val = slice_anchor_data[i];
        if (normalized_anchor_)
          anchor_val *= shape[dim];
        float shape_val = slice_shape_data[i];
        if (normalized_shape_)
          shape_val *= shape[dim];
        int64_t slice_end = static_cast<int64_t>(anchor_val + shape_val);
        DALI_ENFORCE(slice_end <= shape[dim],
          make_string("Slice end for dim", dim, "is out of bounds:",
                      slice_end, ">", shape[dim]));
        slice.anchor[dim] = static_cast<int64_t>(anchor_val);
        slice.shape[dim] = slice_end - slice.anchor[dim];
        assert(slice.anchor[dim] + slice.shape[dim] <= shape[dim]);
      }
      slice.IsInRange(shape);
      return slice;
  };
}

void SliceAttr::VerifyArgsShape(const kernels::TensorShape<>& crop_anchor_shape,
                                const kernels::TensorShape<>& crop_shape_shape) {
  DALI_ENFORCE(crop_anchor_shape == crop_shape_shape);
  size_t args_size = volume(crop_anchor_shape);
  auto dims_size = !dim_names_.empty() ? dim_names_.size() : dims_.size();
  DALI_ENFORCE(args_size == dims_size,
    make_string("Unexpected number of arguments", args_size, "vs", dims_size));
}
*/
}  // namespace dali
