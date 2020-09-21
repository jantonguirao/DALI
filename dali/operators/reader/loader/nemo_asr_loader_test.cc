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
#include "dali/operators/reader/loader/nemo_asr_loader.h"
#include "dali/kernels/signal/downmixing.h"

namespace dali {

static void tempfile(std::string& filename, std::string content = "") {
  int fd = mkstemp(&filename[0]);
  ASSERT_NE(-1, fd);
  if (!content.empty())
    write(fd, content.c_str(), content.size());
  close(fd);
}

TEST(NemoAsrLoaderTest, ParseManifest) {
  std::stringstream ss;
  ss << R"code({"audio_filepath": "path/to/audio1.wav", "duration": 1.45, "text": "     A ab B C D   "})code" << std::endl;
  ss << R"code({"audio_filepath": "path/to/audio2.wav", "duration": 2.45, "offset": 1.03, "text": "C DA B"})code" << std::endl;
  ss << R"code({"audio_filepath": "path/to/audio3.wav", "duration": 3.45})code" << std::endl;
  std::vector<NemoAsrEntry> entries;
  detail::ParseManifest(entries, ss);
  ASSERT_EQ(3, entries.size());

  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
  EXPECT_NEAR(1.45, entries[0].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[0].offset, 1e-7);
  EXPECT_EQ("     A ab B C D   ", entries[0].text);

  EXPECT_EQ("path/to/audio2.wav", entries[1].audio_filepath);
  EXPECT_NEAR(2.45, entries[1].duration, 1e-7);
  EXPECT_NEAR(1.03, entries[1].offset, 1e-7);
  EXPECT_EQ("C DA B", entries[1].text);

  EXPECT_EQ("path/to/audio3.wav", entries[2].audio_filepath);
  EXPECT_NEAR(3.45, entries[2].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[2].offset, 1e-7);
  EXPECT_EQ("", entries[2].text);

  entries.clear();
  ss.clear();
  ss.seekg(0);

  detail::ParseManifest(entries, ss, 0.0f, 0.0f, true);
  ASSERT_EQ(3, entries.size());

  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
  EXPECT_NEAR(1.45, entries[0].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[0].offset, 1e-7);
  EXPECT_EQ("a ab b c d", entries[0].text);

  EXPECT_EQ("path/to/audio2.wav", entries[1].audio_filepath);
  EXPECT_NEAR(2.45, entries[1].duration, 1e-7);
  EXPECT_NEAR(1.03, entries[1].offset, 1e-7);
  EXPECT_EQ("c da b", entries[1].text);

  EXPECT_EQ("path/to/audio3.wav", entries[2].audio_filepath);
  EXPECT_NEAR(3.45, entries[2].duration, 1e-7);
  EXPECT_NEAR(0.0, entries[2].offset, 1e-7);
  EXPECT_EQ("", entries[2].text);

  entries.clear();
  ss.clear();
  ss.seekg(0);

  detail::ParseManifest(entries, ss, 2.0f, 3.0f);  // first and third sample should be ignored
  ASSERT_EQ(1, entries.size());
  EXPECT_EQ("path/to/audio2.wav", entries[0].audio_filepath);

  entries.clear();
  ss.clear();
  ss.seekg(0);
  detail::ParseManifest(entries, ss, 0.5f, 2.45f);  // second sample has a duration of exactly 2.45s
  ASSERT_EQ(2, entries.size());
  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
  EXPECT_EQ("path/to/audio2.wav", entries[1].audio_filepath);

  entries.clear();
  ss.clear();
  ss.seekg(0);
  detail::ParseManifest(entries, ss, 0.0, 2.44999f);
  ASSERT_EQ(1, entries.size());
  EXPECT_EQ("path/to/audio1.wav", entries[0].audio_filepath);
}

TEST(NemoAsrLoaderTest, WrongManifestPath) {
  auto spec = OpSpec("NemoAsrReader")
                  .AddArg("manifest_filepaths", std::vector<std::string>{"./wrong/file.txt"})
                  .AddArg("batch_size", 32)
                  .AddArg("device_id", -1);
  NemoAsrLoader loader(spec);
  ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
}

TEST(NemoAsrLoaderTest, ParseManifestContent) {
  std::string manifest_filepath =
      "/tmp/nemo_asr_manifest_XXXXXX";  // XXXXXX is replaced in tempfile()
  tempfile(manifest_filepath, "{ broken_json ]");

  auto spec = OpSpec("NemoAsrReader")
                  .AddArg("manifest_filepaths", std::vector<std::string>{manifest_filepath})
                  .AddArg("batch_size", 32)
                  .AddArg("device_id", -1);

  {
    NemoAsrLoader loader(spec);
    ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
    EXPECT_EQ(0, loader.Size());
  }

  {
    std::ofstream f(manifest_filepath);
    f << "{}\n{}\n{}";
    f.close();

    NemoAsrLoader loader(spec);
    ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
    EXPECT_EQ(0, loader.Size());
  }

  {
    std::ofstream f(manifest_filepath);
    f << "bla bla bla";
    f.close();

    NemoAsrLoader loader(spec);
    ASSERT_THROW(loader.PrepareMetadata(), std::runtime_error);
    EXPECT_EQ(0, loader.Size());
  }

  {
    std::ofstream f(manifest_filepath);
    f << R"code({"audio_filepath": "/audio/filepath.wav", "text": "this is an example", "duration": 0.32})code";
    f.close();

    NemoAsrLoader loader(spec);
    loader.PrepareMetadata();
    EXPECT_EQ(1, loader.Size());
  }

  ASSERT_EQ(0, remove(manifest_filepath.c_str()));
}

}  // namespace dali

