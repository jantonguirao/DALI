# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(color_manipulation)
add_subdirectory(convolution)
add_subdirectory(jpeg)
add_subdirectory(pointwise)
add_subdirectory(resample)
add_subdirectory(paste)
add_subdirectory(structure)

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_KERNEL_SRCS PARENT_SCOPE)
collect_test_sources(DALI_KERNEL_TEST_SRCS PARENT_SCOPE)
