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

#include "dali/imgcodec/image_decoder.h"
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/stream.h"
#include "dali/imgcodec/decoders/decoder_test_helper.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/test/dali_test_config.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/util/file.h"
#include "dali/util/numpy.h"

namespace dali {
namespace imgcodec {

namespace test {
namespace {
template<typename... Args>
std::string join(Args... args) {
  return make_string_delim('/', args...);
}

std::vector<uint8_t> read_file(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

struct ImageBuffer {
  std::vector<uint8_t> buffer;
  ImageSource src;
  explicit ImageBuffer(const std::string &filename)
  : buffer(read_file(filename))
  , src(ImageSource::FromHostMem(buffer.data(), buffer.size())) {}
};

const auto img_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg");
const auto ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/jpeg");
const auto jpeg_image0 = join(img_dir, "134/site-1534685_1280.jpg");
const auto ref_prefix0 = join(ref_dir, "site-1534685_1280");
}  // namespace

// TEST(ImageDecoderTest, GetInfo) {
//   ImageDecoder dec(CPU_ONLY_DEVICE_ID, true);

//   auto filename = testing::dali_extra_path() + "/db/single/jpeg/100/swan-3584559_640.jpg";
//   ImageSource src = ImageSource::FromFilename(filename);

//   auto info = dec.GetInfo(&src);

//   EXPECT_EQ(info.shape, TensorShape<>(408, 640, 3));
//   EXPECT_FALSE(info.orientation.flip_x);
//   EXPECT_FALSE(info.orientation.flip_y);
//   EXPECT_EQ(info.orientation.rotate, 0);
// }

// TEST(ImageDecoderTest, DecodeToHost_CPU) {
//   int device_id = CPU_ONLY_DEVICE_ID;
//   ImageDecoder dec(device_id, true);

//   auto filename = testing::dali_extra_path() + "/db/single/jpeg/100/swan-3584559_640.jpg";
//   ImageSource src = ImageSource::FromFilename(filename);

//   auto info = dec.GetInfo(&src);

//   ThreadPool tp(4, device_id, false, "ImageDecoderTest");

//   TensorList<CPUBackend> out;
//   out.Resize(uniform_list_shape(1, info.shape), DALI_UINT8);
//   SampleView<CPUBackend> sv = out[0];
//   DecodeResult res = dec.Decode({ &tp, 0 }, sv, &src, {}, {});
//   EXPECT_TRUE(res.success);
//   ASSERT_NO_THROW(
//     if (res.exception)
//       std::rethrow_exception(res.exception);
//   );  // NOLINT
// }

template<typename Backend, typename OutputType>
class ImageDecoderTest : public ::testing::Test {
 public:
  static const auto dtype = type2id<OutputType>::value;

  explicit ImageDecoderTest(int threads_cnt = 4)
      : tp_(threads_cnt, GetDeviceId(), false, "Decoder test")
      , decoder_(GetDeviceId(), false, {}) {}

  /**
  * @brief Decodes an image and returns the result as a CPU tensor.
  */
  TensorView<StorageCPU, const OutputType> Decode(ImageSource *src, const DecodeParams &opts = {},
                                                  const ROI &roi = {}) {
    DecodeContext ctx;
    ctx.tp = &tp_;

    EXPECT_TRUE(decoder_.CanDecode(ctx, src, opts));

    ImageInfo info = decoder_.GetInfo(src);
    auto shape = AdjustToRoi(info.shape, roi);

    // Number of channels can be different than input's due to color conversion
    // TODO(skarpinski) Don't assume channel-last layout here
    *(shape.end() - 1) = NumberOfChannels(opts.format, *(info.shape.end() - 1));

    output_.reshape({{shape}});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tv = output_.cpu()[0];
      SampleView<CPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      DecodeResult decode_result = decoder_.Decode(ctx, view, src, opts, roi);
      EXPECT_TRUE(decode_result.success);
      return tv;
    } else {  // GPU
      auto tv = output_.gpu()[0];
      SampleView<GPUBackend> view(tv.data, tv.shape, type2id<OutputType>::value);
      auto stream_lease = CUDAStreamPool::instance().Get(GetDeviceId());
      ctx.stream = stream_lease;
      auto decode_result = decoder_.Decode(ctx, view, src, opts, roi);
      EXPECT_TRUE(decode_result.success);
      CUDA_CALL(cudaStreamSynchronize(ctx.stream));
      return output_.cpu()[0];
    }
  }

  /**
   * @brief Decodes a batch of images, invoking the batch version of ImageDecoder::Decode
   */
  TensorListView<StorageCPU, const OutputType> Decode(cspan<ImageSource *> in,
                                                      const DecodeParams &opts = {},
                                                      cspan<ROI> rois = {}) {
    int n = in.size();
    std::vector<TensorShape<>> shape(n);

    DecodeContext ctx;
    ctx.tp = &tp_;

    for (int i = 0; i < n; i++) {
      EXPECT_TRUE(decoder_.CanDecode(ctx, in[i], opts));
      ImageInfo info = decoder_.GetInfo(in[i]);
      shape[i] = AdjustToRoi(info.shape, rois.empty() ? ROI{} : rois[i]);
    }

    output_.reshape(TensorListShape{shape});

    if (GetDeviceId() == CPU_ONLY_DEVICE_ID) {
      auto tlv = output_.cpu();
      std::vector<SampleView<CPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto res = decoder_.Decode(ctx, make_span(view), in, opts, rois);
      for (auto decode_result : res)
        EXPECT_TRUE(decode_result.success);
      return tlv;
    } else {  // GPU
      auto tlv = output_.gpu();
      std::vector<SampleView<GPUBackend>> view(n);
      for (int i = 0; i < n; i++)
        view[i] = {tlv[i].data, tlv[i].shape, type2id<OutputType>::value};
      auto stream = CUDAStreamPool::instance().Get(GetDeviceId());
      ctx.stream = stream;
      auto res = decoder_.Decode(ctx, make_span(view), in, opts, rois);
      for (auto decode_result : res)
        EXPECT_TRUE(decode_result.success);
      CUDA_CALL(cudaStreamSynchronize(stream));
      return output_.cpu();
    }
  }

  /**
  * @brief Reads the reference image from specified path and returns it as a tensor.
  */
  Tensor<CPUBackend> ReadReferenceFrom(const std::string &reference_path) {
    auto src = FileStream::Open(reference_path, false, false);
    return numpy::ReadTensor(src.get());
  }

  /**
   * @brief Get device_id for the Backend
   */
  int GetDeviceId() {
    if constexpr (std::is_same<Backend, CPUBackend>::value) {
      return CPU_ONLY_DEVICE_ID;
    } else {
      static_assert(std::is_same<Backend, GPUBackend>::value);
      int device_id;
      CUDA_CALL(cudaGetDevice(&device_id));
      return device_id;
    }
  }

  DecodeParams GetParams() {
    DecodeParams opts{};
    opts.dtype = dtype;
    return opts;
  }

 protected:
  DecodeContext Context() {
    DecodeContext ctx;
    ctx.tp = &tp_;
    return ctx;
  }

 private:
  ThreadPool tp_;  // we want the thread pool to outlive the decoder instance
  ImageDecoder decoder_;
  kernels::TestTensorList<OutputType> output_;
};

template<typename OutputType>
class ImageDecoderTest_CPU : public ImageDecoderTest<CPUBackend, OutputType> {
};

using DecodeOutputTypes = ::testing::Types<uint8_t, int16_t, float>;
TYPED_TEST_SUITE(ImageDecoderTest_CPU, DecodeOutputTypes);

TYPED_TEST(ImageDecoderTest_CPU, DecodeSingleImage) {
  ImageBuffer image(jpeg_image0);
  auto decoded = this->Decode(&image.src, this->GetParams());
  auto ref = this->ReadReferenceFrom(make_string(ref_prefix0, ".npy"));
  AssertEqualSatNorm(decoded, ref);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
