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

#ifndef DALI_OPERATORS_VIDEO_STREAM_CROP_H_
#define DALI_OPERATORS_VIDEO_STREAM_CROP_H_

extern "C" {
#include <libavcodec/avcodec.h>
}
#include <cstdint>
#include <optional>
#include "dali/core/api_helper.h"

namespace dali {

/**
 * @brief The area of the coded frame that a video stream declares as the picture to display.
 *
 * It comes from the cropping rectangle of the H.264 SPS or the conformance window of the HEVC SPS.
 * The size that the container declares may differ from it.
 */
struct StreamCrop {
  int coded_width = 0, coded_height = 0;
  int left = 0, top = 0, right = 0, bottom = 0;

  int width() const {
    return coded_width - left - right;
  }

  int height() const {
    return coded_height - top - bottom;
  }

  /**
   * NVDEC rounds the left and top offset of the display area down to a multiple of 4, so the
   * picture would be shifted if the offsets were not aligned.
   */
  bool IsOriginSupported() const {
    return left % 4 == 0 && top % 4 == 0;
  }

  /**
   * Tells if the coded size that the NVDEC parser reports belongs to the SPS of this crop.
   * The parser may round up the size from the SPS to a multiple of 16, which is what HEVC streams
   * with sizes that are not aligned get.
   */
  bool MatchesCodedSize(unsigned parser_width, unsigned parser_height) const {
    return parser_width >= static_cast<unsigned>(coded_width) &&
           parser_width < static_cast<unsigned>(coded_width) + 16 &&
           parser_height >= static_cast<unsigned>(coded_height) &&
           parser_height < static_cast<unsigned>(coded_height) + 16;
  }

  bool operator==(const StreamCrop &o) const {
    return coded_width == o.coded_width && coded_height == o.coded_height && left == o.left &&
           top == o.top && right == o.right && bottom == o.bottom;
  }
};

/**
 * @brief Reads the display area from the parameter sets in the codec's extradata.
 *
 * Supports H.264 (avcC or Annex B) and HEVC (hvcC or Annex B).
 *
 * @return The display area, or nullopt if the codec is not supported or the extradata does not
 *         contain a valid SPS.
 */
DLL_PUBLIC std::optional<StreamCrop> ParseStreamCrop(AVCodecID codec_id, const uint8_t *extradata,
                                                     int size);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_STREAM_CROP_H_
