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

#ifndef DALI_OPERATORS_READER_NEMO_ASR_READER_OP_H_
#define DALI_OPERATORS_READER_NEMO_ASR_READER_OP_H_

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <istream>
#include <memory>
#include "dali/pipeline/data/views.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"
#include "dali/operators/decoder/audio/generic_decoder.h"
#include "dali/operators/decoder/audio/audio_decoder_impl.h"

namespace dali {

class NemoAsrReader : public DataReader<CPUBackend, NemoAsrEntry> {
 public:
  explicit NemoAsrReader(const OpSpec& spec);

 protected:
  bool CanInferOutputs() const override {
    return false;  // Let RunImpl allocate the outputs
  }

  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  using DecoderType = int16_t;
  struct SampleContext {
    NemoAsrEntry desc;
    AudioMetadata audio_meta;
    std::unique_ptr<AudioDecoderBase> decoder;
  };

  template <typename T>
  void ReadAudio(const TensorView<StorageCPU, T, DynamicDimensions> &audio,
                 SampleContext &sample,
                 Tensor<CPUBackend> &scratch);

  bool read_sr_;
  bool read_text_;

  float sample_rate_;
  float quality_;
  bool downmix_;
  DALIDataType dtype_;
  float max_duration_;
  bool normalize_text_;

  int num_threads_;
  ThreadPool thread_pool_;
  kernels::signal::resampling::Resampler resampler_;
  std::vector<Tensor<CPUBackend>> scratch_;
  std::vector<SampleContext> sample_ctx_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NEMO_ASR_READER_OP_H_
