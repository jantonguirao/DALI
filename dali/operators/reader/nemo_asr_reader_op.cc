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

#include "dali/operators/reader/nemo_asr_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(NemoAsrReader, NemoAsrReader, CPU);

DALI_SCHEMA(NemoAsrReader)
  .NumInput(0)
  .NumOutput(1)
  .DocStr(R"code(Read automatic speech recognition (ASR) data (audio, text) from a 
NVIDIA NeMo compatible manifest.

Example manifest file::

    {"audio_filepath": "path/to/audio1.wav", "duration": 3.45, "text": "this is a nemo tutorial"}
    {"audio_filepath": "path/to/audio1.wav", "offset": 3.45, "duration": 1.45, "text": "same audio file but using offset"}
    {"audio_filepath": "path/to/audio2.wav", "duration": 3.45, "text": "third transcript in this example"}

.. note::
    Only ``audio_filepath`` is field mandatory. If ``duration`` is not specified, the whole audio file will be used. A missing ``text`` field
    will produce an empty string as a text.

.. warning::
    Handing of ``offset`` field is not yet implemented.

This reader produces between 1 and 3 outputs:

- Decoded audio data: float, shape=``(audio_length,)``
- (optional, if ``read_sample_rate=True``) Audio sample rate: float, shape=``(1,)``
- (optional, if ``read_text=True``) Transcript text as a null terminated string: uint8, shape=``(text_len + 1,)``

)code")
  .AddArg("manifest_filepaths",
    "List of paths to NeMo's compatible manifest files.",
    DALI_STRING_VEC)
  .AddOptionalArg("read_sample_rate",
    "Whether to output the sample rate for each sample as a separate output",
    true)
  .AddOptionalArg("read_text",
    "Whether to output the transcript text for each sample as a separate output",
    true)
  .AddOptionalArg("shuffle_after_epoch",
    "If true, reader shuffles whole dataset after each epoch",
    false)
  .AddOptionalArg("sample_rate",
    "If specified, the target sample rate, in Hz, to which the audio is resampled.",
    -1.0f)
  .AddOptionalArg("quality",
    R"code(Resampling quality, 0 is lowest, 100 is highest.

  0 corresponds to 3 lobes of the sinc filter; 50 gives 16 lobes and 100 gives 64 lobes.)code",
     50.0f)
  .AddOptionalArg("downmix",
    "If True, downmix all input channels to mono. "
    "If downmixing is turned on, decoder will produce always 1-D output",
    true)
  .AddOptionalArg("dtype",
    "Type of the output data. Supports types: `INT16`, `INT32`, `FLOAT`",
    DALI_FLOAT)
  .AddOptionalArg("min_duration",
    R"code(It a value greater than 0 is provided, it specifies the minimum allowed duration,
 in seconds, of the audio samples.

Samples with a duration shorter than this value will be ignored.)code",
    0.0f)
  .AddOptionalArg("max_duration",
    R"code(It a value greater than 0 is provided, it specifies the maximum allowed duration,
in seconds, of the audio samples.

Samples with a duration longer than this value will be ignored.)code",
    0.0f)
  .AddOptionalArg("normalize_text",
    "If set to True, the text transcript will be stripped of leading and trailing whitespace "
    "and converted to lowercase.",
    false)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("read_sample_rate"))
         + static_cast<int>(spec.GetArgument<bool>("read_text"));
  })
  .AddParent("LoaderBase");

NemoAsrReader::NemoAsrReader(const OpSpec &spec)
    : DataReader<CPUBackend, NemoAsrEntry>(spec),
      read_sr_(spec.GetArgument<bool>("read_sample_rate")),
      read_text_(spec.GetArgument<bool>("read_text")),
      sample_rate_(spec.GetArgument<float>("sample_rate")),
      quality_(spec.GetArgument<float>("quality")),
      downmix_(spec.GetArgument<bool>("downmix")),
      dtype_(spec.GetArgument<DALIDataType>("dtype")),
      num_threads_(std::max(1, spec.GetArgument<int>("num_threads"))),
      thread_pool_(num_threads_, spec.GetArgument<int>("device_id"), false),
      scratch_(num_threads_) {
  loader_ = InitLoader<NemoAsrLoader>(spec);

  double q = quality_;
  DALI_ENFORCE(q >= 0 && q <= 100, "Resampling quality must be in [0..100] range");
  // this should give 3 lobes for q = 0, 16 lobes for q = 50 and 64 lobes for q = 100
  int lobes = std::round(0.007 * q * q - 0.09 * q + 3);
  resampler_.Initialize(lobes, lobes * 64 + 1);
}

template <typename T>
void NemoAsrReader::ReadAudio(const TensorView<StorageCPU, T, DynamicDimensions> &audio,
                              NemoAsrReader::SampleContext &sample, Tensor<CPUBackend> &scratch) {
  auto &audio_meta = sample.audio_meta;

  bool should_resample = sample_rate_ > 0 && audio_meta.sample_rate != sample_rate_;
  bool should_downmix = audio_meta.channels > 1 && downmix_;

  int64_t decode_scratch_sz = 0;
  int64_t resample_scratch_sz = 0;
  if (should_resample || should_downmix || dtype_ != DALI_INT16)
    decode_scratch_sz = audio_meta.length * audio_meta.channels;

  // resample scratch is used to prepare a single or multiple (depending if
  // downmixing is needed) channel float input, required by the resampling
  // kernel
  int64_t out_channels = should_downmix ? 1 : audio_meta.channels;
  if (should_resample)
    resample_scratch_sz = audio_meta.length * out_channels;

  int64_t total_scratch_sz =
      decode_scratch_sz * sizeof(DecoderType) + resample_scratch_sz * sizeof(float);
  scratch.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
  scratch.Resize({total_scratch_sz});
  uint8_t* scratch_mem = scratch.mutable_data<uint8_t>();

  span<DecoderType> decoder_scratch_mem(reinterpret_cast<DecoderType *>(scratch_mem),
                                        decode_scratch_sz);
  span<float> resample_scratch_mem(
        reinterpret_cast<float *>(scratch_mem + decode_scratch_sz * sizeof(DecoderType)),
        resample_scratch_sz);

  DecodeAudio(audio, *sample.decoder, audio_meta, resampler_,
              decoder_scratch_mem, resample_scratch_mem, sample_rate_, downmix_,
              sample.desc.audio_filepath.c_str());
}

void NemoAsrReader::RunImpl(workspace_t<CPUBackend> &ws) {
  int nsamples = Operator<CPUBackend>::batch_size_;
  int ndim = downmix_ ? 1 : 2;

  sample_ctx_.clear();
  sample_ctx_.resize(nsamples);

  auto &out_audio = ws.OutputRef<CPUBackend>(0);
  out_audio.set_type(TypeTable::GetTypeInfo(dtype_));
  TensorListShape<> out_audio_shape;
  out_audio_shape.resize(nsamples, ndim);

  using Decoder = GenericAudioDecoder<int16_t>;
  for (int i = 0; i < nsamples; i++) {
    auto &sample = sample_ctx_[i];
    sample.desc = GetSample(i);
    sample.decoder = std::make_unique<Decoder>();
    sample.audio_meta = sample.decoder->OpenFromFile(sample.desc.audio_filepath);
    out_audio[i].SetSourceInfo(sample.desc.audio_filepath);
    assert(sample.audio_meta.channels_interleaved);  // it's always true
    auto sample_shape = DecodedAudioShape(sample.audio_meta, sample_rate_, downmix_);
    assert(sample_shape.size() > 0);
    out_audio_shape.set_tensor_shape(i, sample_shape);
  }
  out_audio.Resize(out_audio_shape);

  TYPE_SWITCH(dtype_, type2id, OutType, (float, int16_t), (
    for (int i = 0; i < nsamples; i++) {
      thread_pool_.AddWork([this, &out_audio, i](int tid) {
        auto &sample = sample_ctx_[i];
        ReadAudio<OutType>(view<OutType>(out_audio[i]), sample_ctx_[i], scratch_[tid]);
        sample_ctx_[i].decoder->Close();
      });
    }
  ), DALI_FAIL(make_string("Unsupported type: ", dtype_)));  // NOLINT

  int next_out_idx = 1;
  if (read_sr_) {
    auto &out_sample_rate = ws.OutputRef<CPUBackend>(next_out_idx++);
    out_sample_rate.Resize(uniform_list_shape(nsamples, {1}));
    out_sample_rate.set_type(TypeTable::GetTypeInfo(DALI_FLOAT));
    if (sample_rate_ > 0) {
      for (int i = 0; i < nsamples; i++) {
        out_sample_rate[i].SetSourceInfo(out_audio[i].GetSourceInfo());
        *out_sample_rate[i].mutable_data<float>() = sample_rate_;
      }
    } else {
      for (int i = 0; i < nsamples; i++) {
        out_sample_rate[i].SetSourceInfo(out_audio[i].GetSourceInfo());
        *out_sample_rate[i].mutable_data<float>() = sample_ctx_[i].audio_meta.sample_rate;
      }
    }
  }

  if (read_text_) {
    auto &out_text = ws.OutputRef<CPUBackend>(next_out_idx++);
    out_text.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
    TensorListShape<> out_text_shape;
    out_text_shape.resize(nsamples, 1);
    for (int i = 0; i < nsamples; i++) {
      auto &text = sample_ctx_[i].desc.text;
      int64_t text_sz = text.length() + 1;  // +1 for null character
      out_text_shape.set_tensor_shape(i, {text_sz});
    }
    out_text.Resize(out_text_shape);
    for (int i = 0; i < nsamples; i++) {
      out_text[i].SetSourceInfo(out_audio[i].GetSourceInfo());
      auto &text = sample_ctx_[i].desc.text;
      auto *out_text_sample = out_text[i].mutable_data<uint8_t>();
      std::memcpy(out_text_sample, text.c_str(), text.length());
      out_text_sample[text.length()] = '\0';
    }
  }

  thread_pool_.RunAll();
}


}  // namespace dali
