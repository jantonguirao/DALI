# JpegCompressionDistortion CPU on nvimgcodec — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the `JpegCompressionDistortion` CPU operator to use nvimgcodec for the JPEG encode/decode round-trip, then drop OpenCV's `imgcodecs` from `libdali` (linking it only into test binaries).

**Architecture:** Add a thin `NvImageCodecEncoder` handle wrapper next to the existing `NvImageCodecInstance`/`Decoder`/`CodeStream`/`Image` wrappers in `dali/operators/imgcodec/util/nvimagecodec_types.{h,cc}`. The operator owns one nvimgcodec instance, one encoder, one decoder, lazily constructed on the first `RunImpl`. `RunImpl` flattens (sample, frame), buckets frames by quality, issues one batched `nvimgcodecEncoderEncode` per bucket into per-frame host-memory sinks, then one batched `nvimgcodecDecoderDecode` for the whole batch into the output tensor. `core` and `imgproc` stay on the production link line; `imgcodecs` moves to test-only.

**Tech Stack:** C++17, CMake, nvimgcodec ≥ 0.8.0, libjpeg_turbo_ext (CPU JPEG plugin), DALI's existing `UniqueHandle` template.

**Reference spec:** `docs/superpowers/specs/2026-04-27-jpeg-compression-distortion-nvimgcodec-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `internal_tools/stub_generator/nvimgcodec.json` | Modify | Add encoder/ToHostMem symbols to dynlink allowlist |
| `dali/operators/imgcodec/util/nvimagecodec_types.h` | Modify | Declare `NvImageCodecEncoder`, `NvImageCodecCodeStream::ToHostMem` |
| `dali/operators/imgcodec/util/nvimagecodec_types.cc` | Modify | Implement them |
| `dali/operators/image/distortion/jpeg_compression_distortion_op_cpu.cc` | Modify | Full rewrite — drops OpenCV `imgcodecs`/`imgproc` for codec path |
| `cmake/Dependencies.common.cmake` | Modify | Split production (`core`+`imgproc`) and test-only (`imgcodecs`) OpenCV finds |
| `dali/operators/CMakeLists.txt` | Modify | Link `DALI_OPENCV_TEST_EXTRA_LIBS` into `dali_operator_test` |
| `dali/kernels/CMakeLists.txt` | Modify | Link `DALI_OPENCV_TEST_EXTRA_LIBS` into `dali_kernel_test` |

The encoder wrapper lives in the same file as the existing handle wrappers because (a) it shares the same `UniqueHandle`-based lifetime model, (b) future encoder operators reuse it, (c) anchoring it next to the decoder simplifies discovery.

---

## Build/Test Commands (referenced throughout)

Configure and build (from repo root):

```bash
mkdir -p build && cd build
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TEST=ON \
  -DBUILD_BENCHMARK=OFF \
  -DBUILD_PYTHON=ON
ninja dali dali_kernels dali_operators dali_operator_test dali_kernel_test
```

Re-build only the touched targets after each task:

```bash
ninja -C build dali_operators
```

Run the operator's Python test (the regression gate for Task 3):

```bash
cd build
PYTHONPATH=$PWD/dali/python python -m pytest \
  ../dali/test/python/operator_1/test_jpeg_compression_distortion.py \
  -v
```

Run the C++ operator + kernel gtest binaries (Task 5):

```bash
./build/dali/operators/dali_operator_test.bin --gtest_filter='*Jpeg*'
./build/dali/kernels/dali_kernel_test.bin --gtest_filter='*JpegDistortion*'
```

---

## Task 1: Add encoder symbols to the dynlink_nvimgcodec allowlist

**Files:**
- Modify: `internal_tools/stub_generator/nvimgcodec.json`

**Why first:** without this, every later task that calls an encoder symbol fails to build/link in `WITH_DYNAMIC_NVIMGCODEC=ON` mode (the default). A non-passing build blocks the rest of the plan.

- [ ] **Step 1.1: Edit the JSON allowlist**

Open `internal_tools/stub_generator/nvimgcodec.json` and add four entries inside the `"functions"` object. The existing file is small — preserve exact formatting (two-space indent, trailing-comma style of neighbours).

After: the `"functions"` object should contain these new entries:

```json
"nvimgcodecEncoderCreate": {},
"nvimgcodecEncoderDestroy": {},
"nvimgcodecEncoderEncode": {},
"nvimgcodecCodeStreamCreateToHostMem": {}
```

Place them alphabetically near the other `Codec…`/`Decoder…` entries; ordering is cosmetic only.

- [ ] **Step 1.2: Force regeneration of the dynlink stub**

The custom command in `dali/nvimgcodec/CMakeLists.txt:43-58` already lists this JSON as a dependency, so a clean rebuild of the `dynlink_nvimgcodec` target will regenerate. Touch the JSON to be safe and rebuild only that target:

```bash
touch internal_tools/stub_generator/nvimgcodec.json
ninja -C build dynlink_nvimgcodec
```

Expected: build succeeds, no warnings about unresolved symbols.

- [ ] **Step 1.3: Verify the generated stub contains the new entries**

```bash
grep -E "Encoder(Create|Destroy|Encode)|CodeStreamCreateToHostMem" \
  build/dali/nvimgcodec/dynlink_nvimgcodec_gen.cc | head
```

Expected: 4 lines, one per added symbol.

- [ ] **Step 1.4: Commit**

```bash
git add internal_tools/stub_generator/nvimgcodec.json
git commit -m "Expose nvimgcodec encoder symbols in dynlink_nvimgcodec stub"
```

---

## Task 2: Add `NvImageCodecEncoder` and `CodeStream::ToHostMem` wrappers

**Files:**
- Modify: `dali/operators/imgcodec/util/nvimagecodec_types.h`
- Modify: `dali/operators/imgcodec/util/nvimagecodec_types.cc`

- [ ] **Step 2.1: Add the encoder struct declaration to the header**

In `dali/operators/imgcodec/util/nvimagecodec_types.h`, immediately after the closing brace of `struct DLL_PUBLIC NvImageCodecImage { ... };` (line 135 in the current file) and before the namespace closer, add:

```cpp
struct DLL_PUBLIC NvImageCodecEncoder
    : public UniqueHandle<nvimgcodecEncoder_t, NvImageCodecEncoder> {
  DALI_INHERIT_UNIQUE_HANDLE(nvimgcodecEncoder_t, NvImageCodecEncoder);

  NvImageCodecEncoder() = default;

  static NvImageCodecEncoder Create(nvimgcodecInstance_t instance,
                                    const nvimgcodecExecutionParams_t* exec_params,
                                    const std::string& opts);

  static constexpr nvimgcodecEncoder_t null_handle() {
    return nullptr;
  }

  static void DestroyHandle(nvimgcodecEncoder_t handle);
};
```

- [ ] **Step 2.2: Add the host-memory output sink declaration**

In the same header, inside `struct DLL_PUBLIC NvImageCodecCodeStream { ... }`, add a new static factory after `FromSubCodeStream` and before `null_handle`:

```cpp
  static NvImageCodecCodeStream ToHostMem(nvimgcodecInstance_t instance,
                                          void* ctx,
                                          nvimgcodecResizeBufferFunc_t resize_buffer_func,
                                          const nvimgcodecImageInfo_t* image_info);
```

- [ ] **Step 2.3: Implement the encoder wrapper**

In `dali/operators/imgcodec/util/nvimagecodec_types.cc`, at the end of the `dali::imgcodec` namespace (just before the closing `}  // namespace imgcodec` on line 75), append:

```cpp
NvImageCodecEncoder NvImageCodecEncoder::Create(nvimgcodecInstance_t instance,
                                                const nvimgcodecExecutionParams_t* exec_params,
                                                const std::string& opts) {
  NvImageCodecEncoder ret;
  CHECK_NVIMGCODEC(nvimgcodecEncoderCreate(instance, &ret.handle_, exec_params, opts.c_str()));
  return ret;
}

void NvImageCodecEncoder::DestroyHandle(nvimgcodecEncoder_t handle) {
  nvimgcodecEncoderDestroy(handle);
}
```

- [ ] **Step 2.4: Implement the host-memory output sink**

In the same file, immediately after `NvImageCodecCodeStream::FromSubCodeStream` (which ends at line 56 currently) and before `NvImageCodecCodeStream::DestroyHandle`, add:

```cpp
NvImageCodecCodeStream NvImageCodecCodeStream::ToHostMem(
    nvimgcodecInstance_t instance, void* ctx,
    nvimgcodecResizeBufferFunc_t resize_buffer_func,
    const nvimgcodecImageInfo_t* image_info) {
  NvImageCodecCodeStream ret;
  CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateToHostMem(
      instance, &ret.handle_, ctx, resize_buffer_func, image_info));
  return ret;
}
```

- [ ] **Step 2.5: Build the imgcodec util object**

```bash
ninja -C build dali_operators
```

Expected: no errors. The wrappers are not yet referenced by any caller, so this is a pure compile check (header parses, definitions match).

- [ ] **Step 2.6: Commit**

```bash
git add dali/operators/imgcodec/util/nvimagecodec_types.h \
        dali/operators/imgcodec/util/nvimagecodec_types.cc
git commit -m "Add NvImageCodecEncoder and CodeStream::ToHostMem wrappers"
```

---

## Task 3: Rewrite `JpegCompressionDistortionCPU` to use nvimgcodec

**Files:**
- Modify: `dali/operators/image/distortion/jpeg_compression_distortion_op_cpu.cc`
- Test: `dali/test/python/operator_1/test_jpeg_compression_distortion.py` (existing — used as regression gate, no change)

This is the biggest task. It's not classical TDD because the regression test (the existing Python test) is integration-level, not a unit test for the encoder wrapper. Treat the existing Python test as the failing test until the operator runs end-to-end.

- [ ] **Step 3.1: Replace the entire file body**

Overwrite `dali/operators/image/distortion/jpeg_compression_distortion_op_cpu.cc` with this content:

```cpp
// Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvimgcodec.h>
#include <algorithm>
#include <map>
#include <vector>
#include "dali/operators/image/distortion/jpeg_compression_distortion_op.h"
#include "dali/operators/imgcodec/util/nvimagecodec_types.h"

namespace dali {

DALI_SCHEMA(JpegCompressionDistortion)
    .DocStr(R"code(Introduces JPEG compression artifacts to RGB images.

JPEG is a lossy compression format which exploits characteristics of natural
images and human visual system to achieve high compression ratios. The information
loss originates from sampling the color information at a lower spatial resolution
than the brightness and from representing high frequency components of the image
with a lower effective bit depth. The conversion to frequency domain and quantization
is applied independently to 8x8 pixel blocks, which introduces additional artifacts
at block boundaries.

This operation produces images by subjecting the input to a transformation that
mimics JPEG compression with given `quality` factor followed by decompression.
)code")
    .NumInput(1)
    .InputLayout({"HWC", "FHWC"})
    .NumOutput(1)
    .AddOptionalArg(
        "quality",
        R"code(JPEG compression quality from 1 (lowest quality) to 100 (highest quality).

Any values outside the range 1-100 will be clamped.)code",
        50, true)
    .AllowSequences();

namespace {

using imgcodec::NvImageCodecCodeStream;
using imgcodec::NvImageCodecDecoder;
using imgcodec::NvImageCodecEncoder;
using imgcodec::NvImageCodecImage;
using imgcodec::NvImageCodecInstance;

inline nvimgcodecImageInfo_t MakeRgbU8ImageInfo(uint8_t* buffer, int64_t width, int64_t height) {
  nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                             sizeof(nvimgcodecImageInfo_t), nullptr};
  info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
  info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
  info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
  info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                      sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
  info.num_planes = 1;
  info.plane_info[0].height = static_cast<uint32_t>(height);
  info.plane_info[0].width = static_cast<uint32_t>(width);
  info.plane_info[0].num_channels = 3;
  info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  info.plane_info[0].precision = 8;
  info.plane_info[0].row_stride = static_cast<uint32_t>(width) * 3u;
  info.buffer = buffer;
  info.buffer_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
  info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
  return info;
}

inline nvimgcodecImageInfo_t MakeJpegOutputStreamInfo(int64_t width, int64_t height) {
  nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                             sizeof(nvimgcodecImageInfo_t), nullptr};
  info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
  info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
  info.chroma_subsampling = NVIMGCODEC_SAMPLING_420;
  info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                      sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
  info.num_planes = 1;
  info.plane_info[0].height = static_cast<uint32_t>(height);
  info.plane_info[0].width = static_cast<uint32_t>(width);
  info.plane_info[0].num_channels = 3;
  info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  info.plane_info[0].precision = 8;
  info.plane_info[0].row_stride = static_cast<uint32_t>(width) * 3u;
  std::snprintf(info.codec_name, NVIMGCODEC_MAX_CODEC_NAME_SIZE, "%s", "jpeg");
  return info;
}

unsigned char* ResizeVectorBufferCb(void* ctx, size_t req_size) {
  auto* v = static_cast<std::vector<uint8_t>*>(ctx);
  v->resize(req_size);
  return v->data();
}

}  // namespace

class JpegCompressionDistortionCPU : public JpegCompressionDistortion<CPUBackend> {
 public:
  explicit JpegCompressionDistortionCPU(const OpSpec& spec)
      : JpegCompressionDistortion(spec) {}
  using Operator<CPUBackend>::RunImpl;

 protected:
  void RunImpl(Workspace& ws) override;

 private:
  void EnsureCodecs();

  // Lazily initialized; constructed on first RunImpl invocation.
  bool codecs_ready_ = false;
  imgcodec::NvImageCodecInstance instance_;
  imgcodec::NvImageCodecEncoder encoder_;
  imgcodec::NvImageCodecDecoder decoder_;

  // Reused across RunImpl calls.
  std::vector<std::vector<uint8_t>> encoded_buffers_;
};

void JpegCompressionDistortionCPU::EnsureCodecs() {
  if (codecs_ready_)
    return;

  nvimgcodecInstanceCreateInfo_t instance_create_info{
      NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
  instance_create_info.load_extension_modules = 1;
  instance_create_info.load_builtin_modules = 1;
  instance_create_info.extension_modules_path = nullptr;
  instance_create_info.create_debug_messenger = 1;
  instance_create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL |
                                          NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR |
                                          NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING;
  instance_create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;
  instance_ = imgcodec::NvImageCodecInstance::Create(&instance_create_info);

  nvimgcodecBackend_t cpu_backend{NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
                                  sizeof(nvimgcodecBackend_t), nullptr,
                                  NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
                                  {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
                                   sizeof(nvimgcodecBackendParams_t), nullptr,
                                   1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}};
  nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
                                          sizeof(nvimgcodecExecutionParams_t), nullptr};
  exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
  exec_params.backends = &cpu_backend;
  exec_params.num_backends = 1;
  exec_params.pre_init = 1;
  exec_params.skip_pre_sync = 1;

  encoder_ = imgcodec::NvImageCodecEncoder::Create(instance_, &exec_params, "");
  decoder_ = imgcodec::NvImageCodecDecoder::Create(instance_, &exec_params, "");

  codecs_ready_ = true;
}

void JpegCompressionDistortionCPU::RunImpl(Workspace& ws) {
  EnsureCodecs();

  const auto& input = ws.Input<CPUBackend>(0);
  auto& output = ws.Output<CPUBackend>(0);
  auto layout = input.GetLayout();
  output.SetLayout(layout);
  const auto& in_shape = input.shape();
  const int nsamples = input.num_samples();
  const auto in_view = view<const uint8_t>(input);
  const auto out_view = view<uint8_t>(output);

  struct FrameDesc {
    const uint8_t* in_ptr;
    uint8_t* out_ptr;
    int64_t width;
    int64_t height;
    int quality;
  };
  std::vector<FrameDesc> frames;

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto shape = in_shape.tensor_shape_span(sample_idx);
    int ndim = shape.size();
    int w_dim = layout.find('W');
    int h_dim = layout.find('H');
    int c_dim = layout.find('C');
    assert(w_dim >= 0 && h_dim >= 0 && c_dim >= 0);
    int f_dim = layout.find('F');

    int64_t nframes = volume(shape.begin(), shape.begin() + f_dim + 1);
    int64_t frame_size = volume(shape.begin() + f_dim + 1, shape.begin() + ndim);
    int64_t width = shape[w_dim];
    int64_t height = shape[h_dim];

    int q = std::clamp(quality_arg_[sample_idx].data[0], 1, 100);
    for (int64_t elem = 0; elem < nframes; elem++) {
      frames.push_back(FrameDesc{
          in_view[sample_idx].data + elem * frame_size,
          out_view[sample_idx].data + elem * frame_size,
          width, height, q});
    }
  }

  const size_t N = frames.size();
  encoded_buffers_.clear();
  encoded_buffers_.resize(N);

  // ---- Encode pass: bucket by quality, one batched submit per bucket ----
  std::map<int, std::vector<size_t>> by_quality;
  for (size_t i = 0; i < N; i++)
    by_quality[frames[i].quality].push_back(i);

  for (const auto& kv : by_quality) {
    const int q = kv.first;
    const auto& idxs = kv.second;
    const int batch = static_cast<int>(idxs.size());

    std::vector<imgcodec::NvImageCodecImage> in_imgs(batch);
    std::vector<imgcodec::NvImageCodecCodeStream> out_streams(batch);
    std::vector<nvimgcodecImage_t> in_img_handles(batch);
    std::vector<nvimgcodecCodeStream_t> out_stream_handles(batch);

    for (int k = 0; k < batch; k++) {
      const auto& fd = frames[idxs[k]];
      auto in_info = MakeRgbU8ImageInfo(const_cast<uint8_t*>(fd.in_ptr), fd.width, fd.height);
      in_imgs[k] = imgcodec::NvImageCodecImage::Create(instance_, &in_info);
      in_img_handles[k] = in_imgs[k];

      auto out_info = MakeJpegOutputStreamInfo(fd.width, fd.height);
      out_streams[k] = imgcodec::NvImageCodecCodeStream::ToHostMem(
          instance_, &encoded_buffers_[idxs[k]], &ResizeVectorBufferCb, &out_info);
      out_stream_handles[k] = out_streams[k];
    }

    nvimgcodecJpegEncodeParams_t jpeg_params{NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS,
                                             sizeof(nvimgcodecJpegEncodeParams_t), nullptr,
                                             /*optimized_huffman=*/0};
    nvimgcodecEncodeParams_t enc_params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS,
                                        sizeof(nvimgcodecEncodeParams_t), &jpeg_params,
                                        NVIMGCODEC_QUALITY_TYPE_QUALITY, static_cast<float>(q)};

    nvimgcodecFuture_t future = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecEncoderEncode(encoder_, in_img_handles.data(),
                                             out_stream_handles.data(), batch,
                                             &enc_params, &future));
    CHECK_NVIMGCODEC(nvimgcodecFutureWaitForAll(future));
    size_t status_size = 0;
    CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future, nullptr, &status_size));
    std::vector<nvimgcodecProcessingStatus_t> statuses(status_size);
    CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &status_size));
    nvimgcodecFutureDestroy(future);
    for (size_t k = 0; k < statuses.size(); k++) {
      DALI_ENFORCE(statuses[k] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
                   make_string("nvimgcodec encode failed for frame ", idxs[k],
                               " (status=", static_cast<int>(statuses[k]), ")"));
    }
  }

  // ---- Decode pass: one batched submit for the whole batch ----
  std::vector<imgcodec::NvImageCodecCodeStream> code_streams(N);
  std::vector<imgcodec::NvImageCodecImage> out_imgs(N);
  std::vector<nvimgcodecCodeStream_t> code_stream_handles(N);
  std::vector<nvimgcodecImage_t> out_img_handles(N);

  for (size_t i = 0; i < N; i++) {
    code_streams[i] = imgcodec::NvImageCodecCodeStream::FromHostMem(
        instance_, encoded_buffers_[i].data(), encoded_buffers_[i].size());
    code_stream_handles[i] = code_streams[i];

    auto out_info = MakeRgbU8ImageInfo(frames[i].out_ptr, frames[i].width, frames[i].height);
    out_imgs[i] = imgcodec::NvImageCodecImage::Create(instance_, &out_info);
    out_img_handles[i] = out_imgs[i];
  }

  nvimgcodecDecodeParams_t dec_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS,
                                      sizeof(nvimgcodecDecodeParams_t), nullptr,
                                      /*apply_exif_orientation=*/0,
                                      /*enable_roi=*/0};

  nvimgcodecFuture_t future = nullptr;
  CHECK_NVIMGCODEC(nvimgcodecDecoderDecode(decoder_, code_stream_handles.data(),
                                           out_img_handles.data(), static_cast<int>(N),
                                           &dec_params, &future));
  CHECK_NVIMGCODEC(nvimgcodecFutureWaitForAll(future));
  size_t status_size = 0;
  CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future, nullptr, &status_size));
  std::vector<nvimgcodecProcessingStatus_t> statuses(status_size);
  CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &status_size));
  nvimgcodecFutureDestroy(future);
  for (size_t i = 0; i < statuses.size(); i++) {
    DALI_ENFORCE(statuses[i] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
                 make_string("nvimgcodec decode failed for frame ", i,
                             " (status=", static_cast<int>(statuses[i]), ")"));
  }
}

DALI_REGISTER_OPERATOR(JpegCompressionDistortion, JpegCompressionDistortionCPU, CPU);

}  // namespace dali
```

Notes on intent embedded in the code (for the engineer reading it cold):

- `NVIMGCODEC_SAMPLEFORMAT_I_RGB` + `NVIMGCODEC_COLORSPEC_SRGB` consumes interleaved RGB U8 directly — no `cvtColor` round-trip.
- `NVIMGCODEC_SAMPLING_420` on the encoder output stream matches the default subsampling that OpenCV's `cv::imencode(".jpg", ...)` uses.
- `optimized_huffman = 0` matches the OpenCV default (no `IMWRITE_JPEG_OPTIMIZE`).
- Quality is clamped to `[1, 100]` once per sample (matches the previous schema docs).
- `nvimgcodecFutureWaitForAll` blocks the caller; we then read processing statuses per image and `DALI_ENFORCE`. Status semantics follow the existing decoder pattern.
- The `ws.GetThreadPool()` call is gone on purpose — nvimgcodec's libjpeg_turbo_ext does its own internal multithreading.

- [ ] **Step 3.2: Build the operators target**

```bash
ninja -C build dali_operators
```

Expected: build succeeds. Common failure modes to watch for:

- "undefined reference to `nvimgcodecEncoderCreate`" → Task 1 wasn't applied or you didn't rebuild `dynlink_nvimgcodec`. Re-run Step 1.2.
- "no member named `chroma_subsampling`" → nvimgcodec header is older than 0.8. Check `cmake/Dependencies.common.cmake` `NVIMGCODEC_MIN_VERSION`.

- [ ] **Step 3.3: Run the regression test**

```bash
cd build
PYTHONPATH=$PWD/dali/python python -m pytest \
  ../dali/test/python/operator_1/test_jpeg_compression_distortion.py \
  -v
```

Expected: all tests pass. If a single tolerance-driven test fails, inspect: it is likely a small numerical drift between OpenCV's bundled libjpeg-turbo and nvimgcodec's libjpeg_turbo_ext. Per the spec, regenerate that specific golden rather than widening tolerance globally; do NOT mask correctness regressions.

- [ ] **Step 3.4: Commit**

```bash
git add dali/operators/image/distortion/jpeg_compression_distortion_op_cpu.cc
git commit -m "Rewrite JpegCompressionDistortion CPU on nvimgcodec"
```

---

## Task 4: Drop `imgcodecs` from the production link line

After Task 3, `libdali.so` no longer needs `imgcodecs`, but it is still pulled in by the production `find_package`. This task removes it from production and re-adds it at test scope.

**Files:**
- Modify: `cmake/Dependencies.common.cmake`
- Modify: `dali/operators/CMakeLists.txt`
- Modify: `dali/kernels/CMakeLists.txt`

- [ ] **Step 4.1: Drop `imgcodecs` from the production OpenCV find**

In `cmake/Dependencies.common.cmake`, replace lines 22 and 24 (the two `find_package(OpenCV ...)` calls inside the existing `if (BUILD_OPENCV)` block):

Before:
```cmake
  find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc imgcodecs)
  if(NOT OpenCV_FOUND)
    find_package(OpenCV 3.0 REQUIRED COMPONENTS core imgproc imgcodecs)
  endif()
```

After:
```cmake
  find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc)
  if(NOT OpenCV_FOUND)
    find_package(OpenCV 3.0 REQUIRED COMPONENTS core imgproc)
  endif()
```

- [ ] **Step 4.2: Shrink `DALI_EXCLUDES`**

In the same file, replace the existing `DALI_EXCLUDES` line (line 31 currently):

Before:
```cmake
  list(APPEND DALI_EXCLUDES libopencv_core.a;libopencv_imgproc.a;libopencv_highgui.a;libopencv_imgcodecs.a;liblibwebp.a;libittnotify.a;libpng.a;liblibtiff.a;liblibjasper.a;libIlmImf.a;liblibjpeg-turbo.a)
```

After:
```cmake
  list(APPEND DALI_EXCLUDES libopencv_core.a;libopencv_imgproc.a;libopencv_highgui.a)
```

(`libopencv_highgui.a` is kept because OpenCV's `core` + `imgproc` link transitively to it in some distributions; harmless if unused.)

- [ ] **Step 4.3: Add a test-scope `imgcodecs` find**

Still in `cmake/Dependencies.common.cmake`, locate the existing `if (BUILD_TEST)` block (around line 43, just below the `BUILD_OPENCV` block). Immediately above the `set(BUILD_GTEST ...)` line, add:

```cmake
  # Test-only OpenCV components (used by tests for image I/O of fixture data
  # and diff dumps). Kept out of DALI_LIBS so that libdali doesn't link
  # opencv_imgcodecs and its bundled codec statics.
  find_package(OpenCV 4.0 QUIET COMPONENTS imgcodecs)
  if(NOT OpenCV_FOUND)
    find_package(OpenCV 3.0 REQUIRED COMPONENTS imgcodecs)
  endif()
  set(DALI_OPENCV_TEST_EXTRA_LIBS opencv_imgcodecs CACHE INTERNAL
      "OpenCV imgcodecs target, for test executables only")
```

- [ ] **Step 4.4: Link `DALI_OPENCV_TEST_EXTRA_LIBS` into `dali_operator_test`**

In `dali/operators/CMakeLists.txt`, locate the `dali_operator_test` target's `target_link_libraries` block. After the existing `target_link_libraries(dali_operator_test PRIVATE gtest dynlink_cuda ${DALI_LIBS})` line (currently line 150), add:

```cmake
  if (DALI_OPENCV_TEST_EXTRA_LIBS)
    target_link_libraries(dali_operator_test PRIVATE ${DALI_OPENCV_TEST_EXTRA_LIBS})
  endif()
```

- [ ] **Step 4.5: Link `DALI_OPENCV_TEST_EXTRA_LIBS` into `dali_kernel_test`**

In `dali/kernels/CMakeLists.txt`, locate the `dali_kernel_test` target. After the existing `target_link_libraries(dali_kernel_test PRIVATE gtest dynlink_cuda ${DALI_LIBS})` line (currently line 98), add:

```cmake
  if (DALI_OPENCV_TEST_EXTRA_LIBS)
    target_link_libraries(dali_kernel_test PRIVATE ${DALI_OPENCV_TEST_EXTRA_LIBS})
  endif()
```

- [ ] **Step 4.6: Re-configure CMake**

```bash
cmake --build build --target rebuild_cache 2>/dev/null || \
  (cd build && cmake .. -GNinja \
     -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_TEST=ON \
     -DBUILD_BENCHMARK=OFF \
     -DBUILD_PYTHON=ON)
```

Expected: configure succeeds, two messages "Found OpenCV: ..." (one with `core;imgproc`, one with `imgcodecs`).

- [ ] **Step 4.7: Build everything**

```bash
ninja -C build dali_operators dali_kernels dali dali_operator_test dali_kernel_test
```

Expected: clean build. Common failure: "undefined reference to `cv::imread`" → some test is including OpenCV imgcodecs but the test binary it ends up in did not get the test-extra link. Re-check Step 4.4 and Step 4.5 against the actual target names (look at the existing `target_link_libraries(... ${DALI_LIBS})` lines for the canonical target name to match).

- [ ] **Step 4.8: Verify `libdali.so` no longer references `imgcodecs`**

```bash
nm -D --defined-only build/dali/python/nvidia/dali/libdali.so 2>/dev/null \
  | grep -i imgcodecs | head
```

Expected: no output. If any symbol like `cv::imread` shows up in the *defined* symbols, OpenCV's imgcodecs got statically merged in — investigate `DALI_LIBS` plumbing.

```bash
nm -u build/dali/python/nvidia/dali/libdali.so 2>/dev/null \
  | grep -i imgcodecs | head
```

Expected: no output. If imgcodecs appears as *undefined*, it leaked into the dynamic link list — reverify Step 4.1.

- [ ] **Step 4.9: Run the operator regression test once more**

```bash
cd build
PYTHONPATH=$PWD/dali/python python -m pytest \
  ../dali/test/python/operator_1/test_jpeg_compression_distortion.py \
  -v
```

Expected: still passes (operator path is unchanged from Task 3; this is sanity check for the CMake reshuffle).

- [ ] **Step 4.10: Commit**

```bash
git add cmake/Dependencies.common.cmake \
        dali/operators/CMakeLists.txt \
        dali/kernels/CMakeLists.txt
git commit -m "Drop OpenCV imgcodecs from libdali, link only into test binaries"
```

---

## Task 5: Final verification

**Files:** none

- [ ] **Step 5.1: Run the C++ test binaries**

```bash
./build/dali/operators/dali_operator_test.bin --gtest_filter='*Jpeg*' --gtest_color=yes
./build/dali/kernels/dali_kernel_test.bin --gtest_filter='*Jpeg*:*Resampling*:*Warp*' --gtest_color=yes
```

Expected: all selected tests pass. Failures in Resampling/Warp would indicate the test-scope `imgcodecs` link did not actually reach those compilation units; in that case re-check Step 4.5.

- [ ] **Step 5.2: Run the broader Python operator tests for the distortion area**

```bash
cd build
PYTHONPATH=$PWD/dali/python python -m pytest \
  ../dali/test/python/operator_1/ \
  -v -k "distortion or jpeg"
```

Expected: pass.

- [ ] **Step 5.3: Confirm OpenCV deps shrank in the wheel staging**

```bash
ldd build/dali/python/nvidia/dali/libdali.so | grep -i opencv
```

Expected: only `libopencv_core.so.*` and `libopencv_imgproc.so.*` (and possibly `libopencv_highgui.so.*` from transitive deps). NO `libopencv_imgcodecs.so.*`.

- [ ] **Step 5.4: Done**

If all of the above pass, the change is complete. Bench-mark the operator with the existing TL1 decoder-perf scaffolding before merging upstream (per spec risks), but that is out of scope for this plan.

---

## Self-review checklist

This plan was self-reviewed against the spec. Coverage map:

| Spec section | Tasks covering it |
|---|---|
| New encoder wrapper (`NvImageCodecEncoder`) | Task 2 |
| Host-mem output sink (`CodeStream::ToHostMem`) | Task 2 |
| Operator state members + lazy init | Task 3 |
| RunImpl flow (per-quality bucketed encode + batched decode) | Task 3 |
| Color-space simplification (drop cvtColor) | Task 3 |
| Drop `imgcodecs` from production CMake | Task 4 |
| Test-scope `imgcodecs` find + link | Task 4 |
| `DALI_EXCLUDES` shrink | Task 4.2 |
| Verify `libdali.so` no longer carries `imgcodecs` | Task 4.8, Task 5.3 |
| Operator regression test (Python) | Task 3.3, Task 4.9, Task 5.2 |
| Test-side imgcodecs callers (audit) | Implicit — covered by Task 4.4 + 4.5 + 5.1 |

**Implicit prerequisite added during planning:** Task 1 (dynlink stub allowlist) — not in the spec but required for the dynamic-load build configuration to resolve encoder symbols at runtime.
