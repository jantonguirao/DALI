// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/random/rng_base_cpu.h"
#include "dali/operators/random/uniform_distribution.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

DALI_SCHEMA(random__Uniform)
    .DocStr(R"code(Generates random numbers following a uniform distribution.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a scalar is
generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg("range",
      R"code(Range ``[min, max)`` of a continuous uniform distribution.

This argument is mutually exclusive with ``values``.)code",
      std::vector<float>{-1.0f, 1.0f}, true)
    .AddOptionalArg<std::vector<float>>("values",
      R"code(The discrete values produced by a discrete uniform distribution.

This argument is mutually exclusive with ``range``.)code",
      nullptr, true)
    .AddParent("RNGAttr");

DALI_REGISTER_OPERATOR(random__Uniform, UniformDistribution<CPUBackend>, CPU);

// Deprecated alias
DALI_SCHEMA(Uniform)
    .DocStr(R"code(Generates random numbers following a uniform distribution.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a scalar is
generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg("range",
      R"code(Range ``[min, max)`` of a continuous uniform distribution.

This argument is mutually exclusive with ``values``.)code",
      std::vector<float>{-1.0f, 1.0f}, true)
    .AddOptionalArg<std::vector<float>>("values",
      R"code(The discrete values produced by a discrete uniform distribution.

This argument is mutually exclusive with ``range``.)code",
      nullptr, true)
    .AddParent("RNGAttr")
    .Deprecate("random__Uniform");  // Deprecated in 0.30


DALI_REGISTER_OPERATOR(Uniform, UniformDistribution<CPUBackend>, CPU);

}  // namespace dali
