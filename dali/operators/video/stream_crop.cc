// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/stream_crop.h"
#include <utility>
#include <vector>

namespace dali {

namespace {

// Reads the RBSP of a NAL unit. The emulation prevention bytes are removed.
class BitReader {
 public:
  BitReader(const uint8_t *data, size_t size) {
    data_.reserve(size);
    int zeros = 0;
    for (size_t i = 0; i < size; i++) {
      if (zeros >= 2 && data[i] == 3) {
        zeros = 0;
        continue;
      }
      zeros = data[i] == 0 ? zeros + 1 : 0;
      data_.push_back(data[i]);
    }
  }

  bool ok() const {
    return !error_;
  }

  uint32_t u(int bits) {
    uint32_t v = 0;
    for (int i = 0; i < bits; i++) {
      if (pos_ >= data_.size() * 8) {
        error_ = true;
        return 0;
      }
      v = (v << 1) | ((data_[pos_ >> 3] >> (7 - (pos_ & 7))) & 1);
      pos_++;
    }
    return v;
  }

  void skip(int bits) {
    for (; bits > 0 && !error_; bits -= 32)
      u(bits < 32 ? bits : 32);
  }

  uint32_t ue() {
    int zeros = 0;
    while (!error_ && u(1) == 0) {
      if (++zeros > 31) {
        error_ = true;
        return 0;
      }
    }
    if (error_)
      return 0;
    return (uint32_t(1) << zeros) - 1 + u(zeros);
  }

  int32_t se() {
    uint32_t k = ue();
    int32_t mag = static_cast<int32_t>((k + 1) >> 1);
    return (k & 1) ? mag : -mag;
  }

 private:
  std::vector<uint8_t> data_;
  size_t pos_ = 0;
  bool error_ = false;
};

// Returns false if the list is malformed (delta_scale out of [-128, 127] or truncated data).
bool SkipH264ScalingList(BitReader &br, int size) {
  int last = 8, next = 8;
  for (int j = 0; j < size; j++) {
    if (next != 0) {
      int32_t d = br.se();
      if (!br.ok() || d < -128 || d > 127)
        return false;
      next = (last + d + 256) % 256;
    }
    if (next != 0)
      last = next;
  }
  return br.ok();
}

// Sets the size of the picture and the offsets in the unit of luma samples.
bool Finish(StreamCrop &c, int64_t coded_w, int64_t coded_h, int64_t l, int64_t r, int64_t t,
            int64_t b) {
  const int64_t kMax = 1 << 17;
  if (coded_w <= 0 || coded_h <= 0 || coded_w > kMax || coded_h > kMax)
    return false;
  if (l < 0 || r < 0 || t < 0 || b < 0 || l + r >= coded_w || t + b >= coded_h)
    return false;
  c = {static_cast<int>(coded_w), static_cast<int>(coded_h), static_cast<int>(l),
       static_cast<int>(t), static_cast<int>(r), static_cast<int>(b)};
  return true;
}

// nal points at the NAL header.
std::optional<StreamCrop> ParseH264Sps(const uint8_t *nal, size_t size) {
  if (size < 5)
    return std::nullopt;
  BitReader br(nal + 1, size - 1);
  int profile = br.u(8);
  br.skip(16);  // constraint flags, level
  br.ue();      // sps id
  uint32_t chroma_format_idc = 1;
  uint32_t separate_planes = 0;
  switch (profile) {
    case 100: case 110: case 122: case 244: case 44: case 83: case 86: case 118: case 128:
    case 138: case 139: case 134: case 135:
      chroma_format_idc = br.ue();
      if (chroma_format_idc == 3)
        separate_planes = br.u(1);
      br.ue();      // luma bit depth
      br.ue();      // chroma bit depth
      br.skip(1);   // qpprime_y_zero_transform_bypass_flag
      if (br.u(1)) {
        int n = chroma_format_idc != 3 ? 8 : 12;
        for (int i = 0; i < n && br.ok(); i++) {
          if (br.u(1) && !SkipH264ScalingList(br, i < 6 ? 16 : 64))
            return std::nullopt;
        }
      }
      break;
    default:
      break;
  }
  if (chroma_format_idc > 3)
    return std::nullopt;
  br.ue();  // log2_max_frame_num_minus4
  uint32_t poc_type = br.ue();
  if (poc_type == 0) {
    br.ue();
  } else if (poc_type == 1) {
    br.skip(1);
    br.se();
    br.se();
    uint32_t n = br.ue();
    if (n > 255)
      return std::nullopt;
    for (uint32_t i = 0; i < n && br.ok(); i++)
      br.se();
  } else if (poc_type != 2) {
    return std::nullopt;
  }
  br.ue();      // max_num_ref_frames
  br.skip(1);   // gaps_in_frame_num_value_allowed_flag
  int64_t width_mbs = int64_t(br.ue()) + 1;
  int64_t height_units = int64_t(br.ue()) + 1;
  uint32_t frame_mbs_only = br.u(1);
  if (!frame_mbs_only)
    br.skip(1);  // mb_adaptive_frame_field_flag
  br.skip(1);    // direct_8x8_inference_flag
  int64_t l = 0, r = 0, t = 0, b = 0;
  if (br.u(1)) {
    l = br.ue();
    r = br.ue();
    t = br.ue();
    b = br.ue();
  }
  if (!br.ok())
    return std::nullopt;
  int array_type = separate_planes ? 0 : chroma_format_idc;
  int64_t unit_x = (array_type == 0 || array_type == 3) ? 1 : 2;
  int64_t unit_y = (array_type == 1 ? 2 : 1) * (2 - frame_mbs_only);
  StreamCrop crop;
  if (!Finish(crop, width_mbs * 16, height_units * 16 * (2 - frame_mbs_only), l * unit_x,
              r * unit_x, t * unit_y, b * unit_y))
    return std::nullopt;
  return crop;
}

// nal points at the NAL header (2 bytes).
std::optional<StreamCrop> ParseHevcSps(const uint8_t *nal, size_t size) {
  if (size < 16)
    return std::nullopt;
  BitReader br(nal + 2, size - 2);
  br.skip(4);  // vps id
  uint32_t max_sub_layers_minus1 = br.u(3);
  br.skip(1);   // temporal_id_nesting_flag
  br.skip(96);  // general profile_tier_level
  bool profile_present[8] = {}, level_present[8] = {};
  for (uint32_t i = 0; i < max_sub_layers_minus1; i++) {
    profile_present[i] = br.u(1);
    level_present[i] = br.u(1);
  }
  if (max_sub_layers_minus1 > 0)
    br.skip(2 * (8 - max_sub_layers_minus1));
  for (uint32_t i = 0; i < max_sub_layers_minus1; i++) {
    if (profile_present[i])
      br.skip(88);
    if (level_present[i])
      br.skip(8);
  }
  br.ue();  // sps id
  uint32_t chroma_format_idc = br.ue();
  if (chroma_format_idc > 3)
    return std::nullopt;
  uint32_t separate_planes = chroma_format_idc == 3 ? br.u(1) : 0;
  int64_t w = br.ue();
  int64_t h = br.ue();
  int64_t l = 0, r = 0, t = 0, b = 0;
  if (br.u(1)) {
    l = br.ue();
    r = br.ue();
    t = br.ue();
    b = br.ue();
  }
  if (!br.ok())
    return std::nullopt;
  int array_type = separate_planes ? 0 : chroma_format_idc;
  int64_t unit_x = (array_type == 1 || array_type == 2) ? 2 : 1;
  int64_t unit_y = array_type == 1 ? 2 : 1;
  StreamCrop crop;
  if (!Finish(crop, w, h, l * unit_x, r * unit_x, t * unit_y, b * unit_y))
    return std::nullopt;
  return crop;
}

// Parses all the SPS NAL units and returns the crop, if they all parse and agree with each other.
template <typename ParseSps>
std::optional<StreamCrop> CommonCrop(const std::vector<std::pair<const uint8_t *, size_t>> &spss,
                                     ParseSps &&parse_sps) {
  std::optional<StreamCrop> result;
  for (auto &sps : spss) {
    auto crop = parse_sps(sps.first, sps.second);
    if (!crop)
      return std::nullopt;
    if (result && !(*result == *crop))
      return std::nullopt;
    result = crop;
  }
  return result;
}

// Finds the end of the NAL unit that starts at `start` in an Annex B stream.
int AnnexBNalEnd(const uint8_t *data, int size, int start) {
  int end = start;
  while (end + 2 < size && !(data[end] == 0 && data[end + 1] == 0 && data[end + 2] <= 1))
    end++;
  if (end + 2 >= size)
    end = size;
  return end;
}

std::optional<StreamCrop> ParseH264(const uint8_t *data, int size) {
  std::vector<std::pair<const uint8_t *, size_t>> spss;
  if (size >= 7 && data[0] == 1) {  // avcC
    int num_sps = data[5] & 0x1f;
    int pos = 6;
    for (int i = 0; i < num_sps; i++) {
      if (pos + 2 > size)
        return std::nullopt;
      int len = (data[pos] << 8) | data[pos + 1];
      pos += 2;
      if (len < 1 || pos + len > size)
        return std::nullopt;
      if ((data[pos] & 0x1f) == 7)
        spss.emplace_back(data + pos, len);
      pos += len;
    }
    return CommonCrop(spss, ParseH264Sps);
  }
  // Annex B
  for (int i = 0; i + 3 < size; i++) {
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 && (data[i + 3] & 0x1f) == 7) {
      int start = i + 3;
      int end = AnnexBNalEnd(data, size, start);
      spss.emplace_back(data + start, end - start);
      i = end - 1;
    }
  }
  return CommonCrop(spss, ParseH264Sps);
}

std::optional<StreamCrop> ParseHevc(const uint8_t *data, int size) {
  std::vector<std::pair<const uint8_t *, size_t>> spss;
  if (size >= 23 && data[0] == 1) {  // hvcC
    int num_arrays = data[22];
    int pos = 23;
    for (int a = 0; a < num_arrays; a++) {
      if (pos + 3 > size)
        return std::nullopt;
      int type = data[pos] & 0x3f;
      int num_nalus = (data[pos + 1] << 8) | data[pos + 2];
      pos += 3;
      for (int n = 0; n < num_nalus; n++) {
        if (pos + 2 > size)
          return std::nullopt;
        int len = (data[pos] << 8) | data[pos + 1];
        pos += 2;
        if (len < 2 || pos + len > size)
          return std::nullopt;
        if (type == 33)
          spss.emplace_back(data + pos, len);
        pos += len;
      }
    }
    return CommonCrop(spss, ParseHevcSps);
  }
  for (int i = 0; i + 4 < size; i++) {
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 && ((data[i + 3] >> 1) & 0x3f) == 33) {
      int start = i + 3;
      int end = AnnexBNalEnd(data, size, start);
      spss.emplace_back(data + start, end - start);
      i = end - 1;
    }
  }
  return CommonCrop(spss, ParseHevcSps);
}

}  // namespace

std::optional<StreamCrop> ParseStreamCrop(AVCodecID codec_id, const uint8_t *extradata,
                                          int size) {
  if (!extradata || size <= 0)
    return std::nullopt;
  switch (codec_id) {
    case AV_CODEC_ID_H264:
      return ParseH264(extradata, size);
    case AV_CODEC_ID_HEVC:
      return ParseHevc(extradata, size);
    default:
      return std::nullopt;
  }
}

}  // namespace dali
