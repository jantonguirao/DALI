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

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include "dali/operators/video/stream_crop.h"

namespace dali {
namespace {

class BitWriter {
 public:
  void u(uint32_t v, int bits) {
    for (int i = bits - 1; i >= 0; i--)
      bit((v >> i) & 1);
  }
  void ue(uint32_t v) {
    uint64_t x = uint64_t(v) + 1;
    int len = 0;
    while ((x >> len) > 1)
      len++;
    u(0, len);
    u(static_cast<uint32_t>(x), len + 1);
  }
  void se(int32_t v) {
    ue(v > 0 ? 2 * v - 1 : -2 * v);
  }
  // Appends the trailing bit and returns the bytes with the emulation prevention applied.
  std::vector<uint8_t> Finish() {
    bit(1);
    while (n_ & 7)
      bit(0);
    std::vector<uint8_t> out;
    int zeros = 0;
    for (uint8_t b : bytes_) {
      if (zeros >= 2 && b <= 3) {
        out.push_back(3);
        zeros = 0;
      }
      out.push_back(b);
      zeros = b == 0 ? zeros + 1 : 0;
    }
    return out;
  }

 private:
  void bit(int b) {
    if ((n_ & 7) == 0)
      bytes_.push_back(0);
    bytes_.back() |= b << (7 - (n_ & 7));
    n_++;
  }
  std::vector<uint8_t> bytes_;
  int n_ = 0;
};

struct H264Params {
  int profile = 66;
  int chroma_format_idc = 1;
  bool scaling_lists = false;
  int poc_type = 0;
  int poc_offset = -3;  // offset_for_non_ref_pic, ignored but sets the bits that precede the size
  int width_mbs = 20, height_units = 15;
  bool frame_mbs_only = true;
  int l = 0, r = 0, t = 0, b = 0;
  bool has_crop = true;
};

std::vector<uint8_t> H264Sps(const H264Params &p) {
  BitWriter w;
  w.u(p.profile, 8);
  w.u(0, 8);
  w.u(40, 8);
  w.ue(0);
  if (p.profile >= 100) {
    w.ue(p.chroma_format_idc);
    if (p.chroma_format_idc == 3)
      w.u(0, 1);
    w.ue(0);
    w.ue(0);
    w.u(0, 1);
    w.u(p.scaling_lists, 1);
    if (p.scaling_lists) {
      for (int i = 0; i < (p.chroma_format_idc != 3 ? 8 : 12); i++) {
        bool present = i % 2 == 0;
        w.u(present, 1);
        if (present) {
          int size = i < 6 ? 16 : 64;
          // a few deltas, then a zero next scale which ends the list
          w.se(4);
          w.se(-2);
          if (size > 2)
            w.se(-10);  // 8 + 4 - 2 - 10 = 0
        }
      }
    }
  }
  w.ue(0);
  w.ue(p.poc_type);
  if (p.poc_type == 0) {
    w.ue(4);
  } else if (p.poc_type == 1) {
    w.u(0, 1);
    w.se(p.poc_offset);
    w.se(5);
    w.ue(2);
    w.se(1);
    w.se(-1);
  }
  w.ue(2);
  w.u(0, 1);
  w.ue(p.width_mbs - 1);
  w.ue(p.height_units - 1);
  w.u(p.frame_mbs_only, 1);
  if (!p.frame_mbs_only)
    w.u(0, 1);
  w.u(1, 1);
  w.u(p.has_crop, 1);
  if (p.has_crop) {
    w.ue(p.l);
    w.ue(p.r);
    w.ue(p.t);
    w.ue(p.b);
  }
  w.u(0, 1);  // no VUI
  auto rbsp = w.Finish();
  std::vector<uint8_t> nal = {0x67};
  nal.insert(nal.end(), rbsp.begin(), rbsp.end());
  return nal;
}

std::vector<uint8_t> AvcC(const std::vector<uint8_t> &sps) {
  std::vector<uint8_t> v = {1, 66, 0, 40, 0xff, 0xe1, uint8_t(sps.size() >> 8),
                            uint8_t(sps.size())};
  v.insert(v.end(), sps.begin(), sps.end());
  v.insert(v.end(), {1, 0, 4, 0x68, 1, 2, 3});  // a PPS
  return v;
}

std::vector<uint8_t> AnnexB(const std::vector<uint8_t> &sps) {
  std::vector<uint8_t> v = {0, 0, 0, 1};
  v.insert(v.end(), sps.begin(), sps.end());
  v.insert(v.end(), {0, 0, 0, 1, 0x68, 1, 2, 3});
  return v;
}

std::optional<StreamCrop> Parse(const std::vector<uint8_t> &v, AVCodecID id = AV_CODEC_ID_H264) {
  return ParseStreamCrop(id, v.data(), v.size());
}

}  // namespace

TEST(StreamCropTest, H264NoCrop) {
  H264Params p;
  p.has_crop = false;
  auto c = Parse(AvcC(H264Sps(p)));
  ASSERT_TRUE(c);
  EXPECT_EQ(c->width(), 320);
  EXPECT_EQ(c->height(), 240);
  EXPECT_EQ(c->left, 0);
}

TEST(StreamCropTest, H264Full1080p) {
  H264Params p;
  p.width_mbs = 120;
  p.height_units = 68;
  p.b = 4;
  auto c = Parse(AvcC(H264Sps(p)));
  ASSERT_TRUE(c);
  EXPECT_EQ(c->coded_height, 1088);
  EXPECT_EQ(c->width(), 1920);
  EXPECT_EQ(c->height(), 1080);
}

TEST(StreamCropTest, H264AllSides) {
  H264Params p;
  p.l = 32;
  p.t = 4;
  p.r = 8;
  p.b = 4;
  for (auto data : {AvcC(H264Sps(p)), AnnexB(H264Sps(p))}) {
    auto c = Parse(data);
    ASSERT_TRUE(c);
    EXPECT_EQ(*c, (StreamCrop{320, 240, 64, 8, 16, 8}));
    EXPECT_EQ(c->width(), 240);
    EXPECT_EQ(c->height(), 224);
  }
}

TEST(StreamCropTest, H264Interlaced) {
  H264Params p;
  p.frame_mbs_only = false;
  p.height_units = 8;
  p.b = 2;
  auto c = Parse(AvcC(H264Sps(p)));
  ASSERT_TRUE(c);
  EXPECT_EQ(c->coded_height, 256);
  EXPECT_EQ(c->bottom, 8);
}

TEST(StreamCropTest, H264HighProfile) {
  for (bool lists : {false, true}) {
    for (int chroma : {0, 1, 2, 3}) {
      H264Params p;
      p.profile = 100;
      p.chroma_format_idc = chroma;
      p.scaling_lists = lists;
      p.r = 5;
      p.b = 3;
      auto c = Parse(AvcC(H264Sps(p)));
      ASSERT_TRUE(c) << lists << " " << chroma;
      int ux = (chroma == 0 || chroma == 3) ? 1 : 2;
      int uy = ux;
      EXPECT_EQ(c->right, 5 * ux) << lists << " " << chroma;
      EXPECT_EQ(c->bottom, 3 * uy) << lists << " " << chroma;
    }
  }
}

TEST(StreamCropTest, H264PocType1) {
  H264Params p;
  p.poc_type = 1;
  p.r = 2;
  auto c = Parse(AvcC(H264Sps(p)));
  ASSERT_TRUE(c);
  EXPECT_EQ(c->right, 4);
}

TEST(StreamCropTest, H264EmulationPrevention) {
  // A large ignored value makes a long run of zero bits. Look for the one where it leaves two zero
  // bytes followed by a small one, which need the escape byte. The test fails if the parser does
  // not remove it.
  bool tested = false;
  for (int offset = 1 << 23; offset < (1 << 23) + 4096 && !tested; offset++) {
    H264Params p;
    p.poc_type = 1;
    p.poc_offset = offset;
    auto sps = H264Sps(p);
    bool escaped = false;
    for (size_t i = 0; i + 2 < sps.size(); i++)
      escaped |= sps[i] == 0 && sps[i + 1] == 0 && sps[i + 2] == 3;
    if (!escaped)
      continue;
    auto c = Parse(AvcC(sps));
    ASSERT_TRUE(c);
    EXPECT_EQ(c->coded_width, 320);
    EXPECT_EQ(c->coded_height, 240);
    tested = true;
  }
  EXPECT_TRUE(tested) << "no SPS with an emulation prevention byte was generated";
}

TEST(StreamCropTest, InvalidInput) {
  H264Params p;
  p.r = 200;  // larger than the picture
  EXPECT_FALSE(Parse(AvcC(H264Sps(p))));
  p.r = 2;
  auto avcc = AvcC(H264Sps(p));
  EXPECT_FALSE(Parse({}));
  for (size_t n : {1, 5, 9, 12}) {
    avcc.resize(n);
    EXPECT_FALSE(Parse(avcc)) << n;
  }
  EXPECT_FALSE(Parse({0, 0, 1, 0x67, 0x42}));
  std::vector<uint8_t> garbage(64, 0xff);
  EXPECT_FALSE(Parse(garbage));
  EXPECT_FALSE(Parse(AvcC(H264Sps(p)), AV_CODEC_ID_VP9));
}

namespace {

std::vector<uint8_t> HevcSps(int chroma, int w, int h, int l, int r, int t, int b,
                             int sub_layers_minus1 = 0) {
  BitWriter bw;
  bw.u(0, 4);
  bw.u(sub_layers_minus1, 3);
  bw.u(1, 1);
  bw.u(0, 32);  // general profile_tier_level, 96 bits
  bw.u(0, 32);
  bw.u(0, 32);
  for (int i = 0; i < sub_layers_minus1; i++) {
    bw.u(i == 0 ? 1 : 0, 1);  // profile present
    bw.u(1, 1);               // level present
  }
  if (sub_layers_minus1 > 0)
    bw.u(0, 2 * (8 - sub_layers_minus1));
  for (int i = 0; i < sub_layers_minus1; i++) {
    if (i == 0) {
      bw.u(0, 32);
      bw.u(0, 32);
      bw.u(0, 24);
    }
    bw.u(0, 8);
  }
  bw.ue(0);
  bw.ue(chroma);
  if (chroma == 3)
    bw.u(0, 1);
  bw.ue(w);
  bw.ue(h);
  bool crop = l || r || t || b;
  bw.u(crop, 1);
  if (crop) {
    bw.ue(l);
    bw.ue(r);
    bw.ue(t);
    bw.ue(b);
  }
  bw.u(0, 1);
  auto rbsp = bw.Finish();
  std::vector<uint8_t> nal = {0x42, 0x01};
  nal.insert(nal.end(), rbsp.begin(), rbsp.end());
  return nal;
}

std::vector<uint8_t> HvcC(const std::vector<uint8_t> &sps) {
  std::vector<uint8_t> v(22, 0);
  v[0] = 1;
  v.push_back(2);                                  // two arrays
  v.insert(v.end(), {32, 0, 1, 0, 2, 0x40, 1});    // a VPS
  v.insert(v.end(), {33, 0, 1, uint8_t(sps.size() >> 8), uint8_t(sps.size())});
  v.insert(v.end(), sps.begin(), sps.end());
  return v;
}

}  // namespace

TEST(StreamCropTest, Hevc) {
  for (int layers : {0, 2}) {
    auto sps = HevcSps(1, 1920, 1088, 0, 0, 0, 4, layers);
    for (auto data : {HvcC(sps)}) {
      auto c = Parse(data, AV_CODEC_ID_HEVC);
      ASSERT_TRUE(c) << layers;
      EXPECT_EQ(*c, (StreamCrop{1920, 1088, 0, 0, 0, 8})) << layers;
    }
    std::vector<uint8_t> annexb = {0, 0, 0, 1};
    annexb.insert(annexb.end(), sps.begin(), sps.end());
    auto c = Parse(annexb, AV_CODEC_ID_HEVC);
    ASSERT_TRUE(c) << layers;
    EXPECT_EQ(c->height(), 1080);
  }
}

TEST(StreamCropTest, HevcChromaUnits) {
  auto c = Parse(HvcC(HevcSps(2, 320, 240, 3, 2, 1, 1)), AV_CODEC_ID_HEVC);
  ASSERT_TRUE(c);
  EXPECT_EQ(*c, (StreamCrop{320, 240, 6, 1, 4, 1}));
  c = Parse(HvcC(HevcSps(3, 320, 240, 3, 2, 1, 1)), AV_CODEC_ID_HEVC);
  ASSERT_TRUE(c);
  EXPECT_EQ(*c, (StreamCrop{320, 240, 3, 1, 2, 1}));
  EXPECT_FALSE(Parse(HvcC(HevcSps(1, 320, 240, 200, 200, 0, 0)), AV_CODEC_ID_HEVC));
}

}  // namespace dali
