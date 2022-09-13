// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include "dali/imgcodec/image_decoder.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {
namespace imgcodec {

TEST(ImageDecoderTest, GetInfo) {
  ImageDecoder dec(CPU_ONLY_DEVICE_ID, true);

  auto filename = testing::dali_extra_path() + "/db/single/jpeg/100/swan-3584559_640.jpg";
  ImageSource src = ImageSource::FromFilename(filename);

  auto info = dec.GetInfo(&src);

  EXPECT_EQ(info.shape, TensorShape<>(408, 640, 3));
  EXPECT_FALSE(info.orientation.flip_x);
  EXPECT_FALSE(info.orientation.flip_y);
  EXPECT_EQ(info.orientation.rotate, 0);
}

TEST(ImageDecoderTest, DecodeToHost_CPU) {
  int device_id = CPU_ONLY_DEVICE_ID;
  ImageDecoder dec(device_id, true);

  auto filename = testing::dali_extra_path() + "/db/single/jpeg/100/swan-3584559_640.jpg";
  ImageSource src = ImageSource::FromFilename(filename);

  auto info = dec.GetInfo(&src);

  ThreadPool tp(4, device_id, false, "ImageDecoderTest");

  TensorList<CPUBackend> out;
  out.Resize(uniform_list_shape(1, info.shape), DALI_UINT8);
  SampleView<CPUBackend> sv = out[0];
  DecodeResult res = dec.Decode({ &tp, 0 }, sv, &src, {}, {});
  EXPECT_TRUE(res.success);
  ASSERT_NO_THROW(
    if (res.exception)
      std::rethrow_exception(res.exception);
  );  // NOLINT
}

}  // namespace imgcodec
}  // namespace dali
