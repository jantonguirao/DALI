// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <sstream>
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/operators/reader/nemo_asr_reader_op.h"
#include "dali/kernels/signal/downmixing.h"
#include "dali/pipeline/pipeline.h"

namespace dali {

static std::string audio_data_root = make_string(testing::dali_extra_path(), "/db/audio/wav/");  // NOLINT

static void tempfile(std::string& filename, std::string content = "") {
  int fd = mkstemp(&filename[0]);
  ASSERT_NE(-1, fd);
  if (!content.empty())
    write(fd, content.c_str(), content.size());
  close(fd);
}

TEST(NemoAsrReaderTest, ReadSample) {
  std::string manifest_filepath =
      "/tmp/nemo_asr_manifest_XXXXXX";  // XXXXXX is replaced in tempfile()

  tempfile(manifest_filepath);
  std::ofstream f(manifest_filepath);
  f << "{\"audio_filepath\": \"" << make_string(audio_data_root, "dziendobry.wav") << "\""
    << ", \"text\": \"dzien dobry\""
    << ", \"duration\": 3.0"
    << "}";
  f.close();

  // Contains wav file to be decoded
  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  // Contains raw PCM data decoded offline
  std::string decoded_path = make_string(audio_data_root, "dziendobry.txt");

  std::ifstream file(decoded_path.c_str());
  std::vector<int16_t> ref_data{std::istream_iterator<int16_t>(file),
                                std::istream_iterator<int16_t>()};
  int64_t ref_sz = ref_data.size();
  int64_t ref_samples = ref_sz/2;
  int64_t ref_channels = 2;

  constexpr int batch_size = 8;
  constexpr int num_threads = 3;

  {
    auto spec = OpSpec("NemoAsrReader")
                    .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                    .AddArg("downmix", false)
                    .AddArg("dtype", DALI_INT16)
                    .AddOutput("audio", "cpu")
                    .AddOutput("sample_rate", "cpu")
                    .AddOutput("text", "cpu");

    Pipeline pipe(batch_size, num_threads, 0);
    pipe.AddOperator(spec);
    pipe.Build({{"audio", "cpu"}});
    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    TensorView<StorageCPU, int16_t> ref(ref_data.data(), {ref_samples, 2});
    Check(ref, view<const int16_t>(ws.OutputRef<CPUBackend>(0))[0]);
  }

  std::vector<float> downmixed(ref_samples, 0.0f);
  for (int i = 0; i < ref_samples; i++) {
    double l = ConvertSatNorm<float>(ref_data[2*i]);
    double r = ConvertSatNorm<float>(ref_data[2*i+1]);
    downmixed[i] = (l + r) / 2;
  }
  {
    auto spec = OpSpec("NemoAsrReader")
                    .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                    .AddArg("downmix", true)
                    .AddArg("dtype", DALI_FLOAT)
                    .AddOutput("audio", "cpu")
                    .AddOutput("sample_rate", "cpu")
                    .AddOutput("text", "cpu");

    Pipeline pipe(batch_size, num_threads, 0);
    pipe.AddOperator(spec);
    pipe.Build({{"audio", "cpu"}});
    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    TensorView<StorageCPU, float> ref(downmixed.data(), {ref_samples});
    Check(ref, view<const float>(ws.OutputRef<CPUBackend>(0))[0]);
  }

  {
    float sr_in = 44100.0f;
    float sr_out = 22050.0f;

    DeviceWorkspace ws1;
    {
      auto spec = OpSpec("NemoAsrReader")
                      .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                      .AddArg("downmix", true)
                      .AddArg("sample_rate", sr_out)
                      .AddArg("dtype", DALI_FLOAT)
                      .AddOutput("audio", "cpu")
                      .AddOutput("sample_rate", "cpu")
                      .AddOutput("text", "cpu");

      Pipeline pipe(batch_size, num_threads, 0);
      pipe.AddOperator(spec);
      pipe.Build({{"audio", "cpu"}});
      pipe.RunCPU();
      pipe.RunGPU();
      pipe.Outputs(&ws1);
    }

    int64_t downsampled_len =
        kernels::signal::resampling::resampled_length(ref_samples, sr_in, sr_out);
    std::vector<float> downsampled(downsampled_len, 0.0f);
    constexpr double q = 50.0;
    int lobes = std::round(0.007 * q * q - 0.09 * q + 3);
    kernels::signal::resampling::Resampler resampler;
    resampler.Initialize(lobes, lobes * 64 + 1);
    resampler.Resample(downsampled.data(), 0, downsampled_len, sr_out, downmixed.data(),
                       downmixed.size(), sr_in, 1);

    TensorView<StorageCPU, float> ref(downsampled.data(), {downsampled_len});
    auto view1 = view<const float>(ws1.OutputRef<CPUBackend>(0))[0];
    Check(ref, view1, EqualEpsRel(1e-6, 1e-6));

    DeviceWorkspace ws2;
    {
      auto spec = OpSpec("NemoAsrReader")
                    .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                    .AddArg("downmix", true)
                    .AddArg("sample_rate", static_cast<float>(sr_out))
                    .AddArg("dtype", DALI_INT16)
                    .AddOutput("audio", "cpu")
                    .AddOutput("sample_rate", "cpu")
                    .AddOutput("text", "cpu");

      Pipeline pipe(batch_size, num_threads, 0);
      pipe.AddOperator(spec);
      pipe.Build({{"audio", "cpu"}});
      pipe.RunCPU();
      pipe.RunGPU();
      pipe.Outputs(&ws2);
    }

    auto view2 = view<const int16_t>(ws2.OutputRef<CPUBackend>(0))[0];

    ASSERT_EQ(volume(view1.shape), volume(view2.shape));
    std::vector<float> converted(downsampled_len, 0.0f);
    for (size_t i = 0; i < converted.size(); i++)
      converted[i] = ConvertSatNorm<float>(view2.data[i]);
    TensorView<StorageCPU, float> converted_from_int16(converted.data(), {downsampled_len});
    Check(ref, converted_from_int16, EqualEpsRel(1e-6, 1e-6));
  }
}

}  // namespace dali

