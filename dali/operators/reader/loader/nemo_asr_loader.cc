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

#include <string>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/pipeline/data/views.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/util/lookahead_parser.h"
#include "dali/util/file.h"

namespace dali {

namespace detail {

std::string trim(const std::string& str,
                 const std::string& whitespace = " \t") {
  const auto str_begin = str.find_first_not_of(whitespace);
  if (str_begin == std::string::npos)
    return {};  // no content
  const auto str_end = str.find_last_not_of(whitespace);
  const auto str_len = str_end - str_begin + 1;
  return str.substr(str_begin, str_len);
}

std::string NormalizeText(std::string& text) {
  // Remove trailing and leading whitespace
  auto norm_text = trim(text);

  // Convert to lowercase
  for (auto &c : norm_text) {
    c = std::tolower(c);
  }
  return norm_text;
}

void ParseManifest(std::vector<NemoAsrEntry> &entries, std::istream& manifest_file,
                   float min_duration, float max_duration, bool normalize_text) {
  std::string line;
  while (std::getline(manifest_file, line)) {
    detail::LookaheadParser parser(const_cast<char*>(line.c_str()));
    if (parser.PeekType() != kObjectType) {
      DALI_WARN(make_string("Skipping invalid manifest line: ", line));
      continue;
    }
    parser.EnterObject();
    NemoAsrEntry entry;
    while (const char* key = parser.NextObjectKey()) {
      if (0 == detail::safe_strcmp(key, "audio_filepath")) {
        entry.audio_filepath = parser.GetString();
      } else if (0 == detail::safe_strcmp(key, "duration")) {
        entry.duration = parser.GetDouble();
      } else if (0 == detail::safe_strcmp(key, "offset")) {
        entry.offset = parser.GetDouble();
        DALI_WARN("Handing of ``offset`` is not yet implemented and will be ignored.");
      } else if (0 == detail::safe_strcmp(key, "text")) {
        entry.text = parser.GetString();
        if (normalize_text)
          entry.text = NormalizeText(entry.text);
      } else {
        parser.SkipValue();
      }
    }
    if (entry.audio_filepath.empty()) {
      DALI_WARN(make_string("Skipping manifest line without an audio filepath: ", line));
      continue;
    }

    if ((max_duration > 0.0f && entry.duration > max_duration) ||
        (min_duration > 0.0f && entry.duration < min_duration)) {
      continue;  // skipping sample
    }

    entries.emplace_back(std::move(entry));
  }
}

}  // namespace detail

void NemoAsrLoader::PrepareMetadataImpl() {
  for (auto &manifest_filepath : manifest_filepaths_) {
    std::ifstream fstream(manifest_filepath);
    DALI_ENFORCE(fstream,
                 make_string("Could not open NEMO ASR manifest file: \"", manifest_filepath, "\""));
    detail::ParseManifest(entries_, fstream, max_duration_, normalize_text_);
  }

  DALI_ENFORCE(Size() > 0, "No files found.");
  if (shuffle_) {
    // seeded with hardcoded value to get
    // the same sequence on every shard
    std::mt19937 g(kDaliDataloaderSeed);
    std::shuffle(entries_.begin(), entries_.end(), g);
  }
  Reset(true);
}

void NemoAsrLoader::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, Size()) : 0;
  current_epoch_++;

  if (shuffle_after_epoch_) {
    std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
    std::shuffle(entries_.begin(), entries_.end(), g);
  }
}

void NemoAsrLoader::PrepareEmpty(NemoAsrEntry &sample) {
  sample = {};
}

void NemoAsrLoader::ReadSample(NemoAsrEntry& sample) {
  sample = entries_[current_index_++];
  MoveToNextShard(current_index_);  // handle wrap-around
}

Index NemoAsrLoader::SizeImpl() {
  return entries_.size();
}
}  // namespace dali
