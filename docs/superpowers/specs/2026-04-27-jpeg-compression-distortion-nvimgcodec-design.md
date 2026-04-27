# JpegCompressionDistortion CPU: replace OpenCV imgcodecs with nvimgcodec

Date: 2026-04-27
Status: Approved (awaiting written-spec review)
Author: Joaquin Anton Guirao

## Goal

Eliminate the last production caller of OpenCV's `imgcodecs` library by rewriting
the `JpegCompressionDistortion` CPU operator to perform the JPEG encode/decode
round-trip via nvimgcodec. Once this lands, `imgcodecs` and its bundled codec
statics (libwebp, libpng, libtiff, libjasper, IlmImf, libjpeg-turbo, ittnotify)
can be dropped from `DALI_LIBS`, slimming the wheel.

## Non-goals

- No change to the operator's GPU implementation.
- No change to the operator's schema or per-sample arguments.
- No change to nvimgcodec usage in any other operator.
- No introduction of a direct libjpeg-turbo dependency in DALI's CPU operators
  outside of what nvimgcodec pulls in via its libjpeg_turbo_ext extension.

## Background

The CPU operator currently performs, per frame:

```
RGB input -> cvtColor RGB->BGR -> cv::imencode(".jpg", quality) -> std::vector<uint8_t>
                                                                       |
                                       cv::imdecode(IMREAD_COLOR) <----+
                                                |
                                                v
                                       BGR -> cvtColor BGR->RGB -> RGB output
```

This is the only production call site for `cv::imencode`/`cv::imdecode` in
DALI; every other `imread`/`imwrite`/`imdecode` reference lives in test code
(`*_test.cc`, `dali/test/`, `decoder_test_helper.h`).

DALI today uses nvimgcodec for image decoding (`dali/operators/imgcodec/`) but
has no encoder integration. The C API exposes
`nvimgcodecEncoderCreate/Encode/Destroy`, `nvimgcodecJpegEncodeParams_t`
(`optimized_huffman` flag), and `nvimgcodecEncodeParams_t` with quality types
including `NVIMGCODEC_QUALITY_TYPE_QUALITY` for the conventional 1-100 scale.
Output bitstreams can be written to a host-resizable buffer via
`nvimgcodecCodeStreamCreateToHostMem` with a `nvimgcodecResizeBufferFunc_t`
callback.

`BUILD_OPENCV` and `BUILD_NVIMAGECODEC` are both gated by
`NOT BUILD_DALI_NODEPS`, so the operator's compile-time availability story is
unchanged.

## Design

### New encoder wrapper

Add `NvImageCodecEncoder` to `dali/operators/imgcodec/util/nvimagecodec_types.h`,
mirroring the existing `NvImageCodecDecoder`:

```cpp
struct DLL_PUBLIC NvImageCodecEncoder
    : public UniqueHandle<nvimgcodecEncoder_t, NvImageCodecEncoder> {
  DALI_INHERIT_UNIQUE_HANDLE(nvimgcodecEncoder_t, NvImageCodecEncoder);
  NvImageCodecEncoder() = default;
  static NvImageCodecEncoder Create(nvimgcodecInstance_t instance,
                                    const nvimgcodecExecutionParams_t* exec_params,
                                    const std::string& opts);
  static constexpr nvimgcodecEncoder_t null_handle() { return nullptr; }
  static void DestroyHandle(nvimgcodecEncoder_t handle);
};
```

Plus a host-memory output sink helper on `NvImageCodecCodeStream`:

```cpp
static NvImageCodecCodeStream ToHostMem(
    nvimgcodecInstance_t instance,
    void* ctx, nvimgcodecResizeBufferFunc_t resize_cb,
    const nvimgcodecImageInfo_t* info);
```

Implementation in `nvimagecodec_types.cc` calls
`nvimgcodecEncoderCreate`/`nvimgcodecEncoderDestroy` and
`nvimgcodecCodeStreamCreateToHostMem` respectively. No DLL exports beyond what
the existing wrappers already do.

### Operator state

`JpegCompressionDistortionCPU` gains four members, lazily constructed on the
first `RunImpl`:

- `imgcodec::NvImageCodecInstance instance_`
- `imgcodec::NvImageCodecEncoder encoder_`
- `imgcodec::NvImageCodecDecoder decoder_`
- `std::vector<std::vector<uint8_t>> encoded_buffers_` — one bitstream per frame,
  reused across `RunImpl` calls (resized each call to the current frame count).

Execution params force CPU-only:

```cpp
nvimgcodecExecutionParams_t exec{...};
exec.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
exec.num_backends = 1;
nvimgcodecBackend_t cpu_backend{...};
cpu_backend.kind = NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
exec.backends = &cpu_backend;
```

This selects libjpeg_turbo_ext even when GPU JPEG extensions are loaded, so the
operator's behavior is host-deterministic regardless of GPU presence.

### `RunImpl` flow (single batched submission per quality bucket)

```
flatten (sample, frame) -> frames[N] with (input_ptr, output_ptr, w, h, quality)
encoded_buffers_.resize(N)

# Encode pass: bucket by quality
groups = {}
for i in 0..N-1:
    groups[clamp(frames[i].quality, 1, 100)].append(i)

for (q, idxs) in groups:
    in_imgs[]  = NvImageCodecImage built over frames[idxs[k]].input_ptr
                  with HxW interleaved RGB U8 (sample_format = I_RGB,
                  buffer_kind = HOST)
    out_streams[] = NvImageCodecCodeStream::ToHostMem with resize_cb pointing
                    at encoded_buffers_[idxs[k]] (uses lambda capturing
                    &encoded_buffers_[idx])
    encode_params.quality_type  = NVIMGCODEC_QUALITY_TYPE_QUALITY
    encode_params.quality_value = float(q)
    jpeg_params.optimized_huffman = 0
    encode_params.struct_next   = &jpeg_params
    nvimgcodecEncoderEncode(encoder_, in_imgs, out_streams, len(idxs),
                            &encode_params, &future)
    future.wait()
    for k: assert image[k].processing_status == SUCCESS

# Decode pass: one call for the whole batch
for i in 0..N-1:
    code_streams[i] = NvImageCodecCodeStream::FromHostMem(encoded_buffers_[i])
    out_imgs[i]     = NvImageCodecImage over frames[i].output_ptr (I_RGB U8 HOST)
nvimgcodecDecoderDecode(decoder_, code_streams, out_imgs, N,
                        &decode_params, &future)
future.wait()
for i: assert image[i].processing_status == SUCCESS
```

The decode pass needs no per-quality grouping — quality is purely an encoder
input.

The thread pool (`ws.GetThreadPool()`) is no longer used inside this operator.
nvimgcodec's libjpeg_turbo_ext does its own internal multithreading, and its
batched API already targets parallel execution.

### Color-space change

`I_RGB` is consumed and produced directly. The two `cv::cvtColor` calls
disappear. The encoded JPEG should be byte-identical to the previous version
modulo libjpeg-turbo version drift between the OpenCV-bundled copy and
nvimgcodec's extension copy. Reference image goldens in the existing test
suite tolerate this drift today (the operator already has small differences
across builds).

### CMake changes

`cmake/Dependencies.common.cmake`:

- Line 22: drop `imgcodecs` from production `find_package`:
  `find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc)`.
- Line 24: same for the 3.0 fallback.
- Line 29: `${OpenCV_LIBRARIES}` after this find contains only core+imgproc.
  Cache it under a stable name (e.g. `DALI_OPENCV_PROD_LIBS`) immediately after
  the call so a later imgcodecs find doesn't mutate it.
- Line 31: shrink `DALI_EXCLUDES` — remove `libopencv_imgcodecs.a`,
  `liblibwebp.a`, `libittnotify.a`, `libpng.a`, `liblibtiff.a`, `liblibjasper.a`,
  `libIlmImf.a`, `liblibjpeg-turbo.a`. These statics live inside the imgcodecs
  module; once it's not linked into `libdali`, the excludes are dead.

Test-scope OpenCV link:

- Inside the existing `if (BUILD_TEST)` block in `Dependencies.common.cmake`,
  do a second `find_package(OpenCV ... COMPONENTS imgcodecs)`, then capture the
  newly-resolved imgcodecs target into a stable variable, e.g.
  `set(DALI_OPENCV_TEST_EXTRA_LIBS opencv_imgcodecs)` (or whatever symbolic
  name CMake produces for the imported target).
- Append `${DALI_OPENCV_TEST_EXTRA_LIBS}` privately to the two test executable
  targets:
  - `dali_operator_test` (`dali/operators/CMakeLists.txt:149`)
  - `dali_kernel_test` (`dali/kernels/CMakeLists.txt:92`)
  Do *not* add it to `DALI_LIBS`, since `DALI_LIBS` is consumed by production
  `dali_operators`/`dali_kernels` targets too.
- The kernel test sources from `warp_test/` and `resampling_test/` are
  compiled into the single `dali_kernel_test` executable (the per-directory
  CMakeLists in those folders feed sources upward, they don't define
  separate executables), so the link added above covers them.

`dali/python/bundle-wheel.sh` does not list any OpenCV libs — OpenCV statics
are absorbed into `libdali.so` via `DALI_LIBS` at link time. No bundle changes.

#### Test-side imgcodecs callers (audit)

No test source changes are needed. For reference, the call sites that depend
on test-scope `imgcodecs` are:

- Reference-image loaders (`cv::imread`):
  - `dali/kernels/test/warp_test/warp_cpu_test.cc`
  - `dali/kernels/test/warp_test/warp_gpu_test.cu`
  - `dali/kernels/test/test_data_test.cc`
  - `dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_test.cu`
  - `dali/test/mat2tensor_test.cc`
  - `dali/operators/sequence/optical_flow/optical_flow_test.cc`
  - `dali/operators/video/video_test.h`
  - `dali/operators/reader/coco_reader_op_test.cc`
- In-memory JPEG round-trip references (`cv::imdecode`/`cv::imencode`):
  - `dali/test/dali_test.h` (legacy decoder test base class — used widely)
  - `dali/test/dali_test_decoder.h`
  - `dali/operators/imgcodec/decoder_test_helper.h`
  - `dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_test.cu` (kernel-level
    analogue of the operator we're rewriting; uses OpenCV as the *kernel*
    reference, separate from the operator test)
- Diff-image dump helpers (`cv::imwrite`):
  - `dali/test/dump_diff.h`, `dali/test/cv_mat_utils.h`,
    `dali/test/dali_test_single_op.h`
  - `dali/operators/imgcodec/decoder_test_helper.h`
  - `dali/kernels/test/resampling_test/{resampling_compare_test.cc,
    resampling_internal_test.cu, separable_cpu_test.cc, separable_impl_test.cc}`

### Error handling

- `EncoderCreate`/`DecoderCreate` failure → throw on first `RunImpl` (matches
  the decoder operator pattern). `CHECK_NVIMGCODEC` is the existing macro.
- Per-image `processing_status != SUCCESS` after future wait → throw
  `DALIException` reporting the frame index and the integer status code.
- The `nvimgcodecResizeBufferFunc_t` callback is invoked from inside
  `nvimgcodecEncoderEncode` on internal worker threads; it must be reentrant
  per buffer. Each frame's callback writes to its own
  `encoded_buffers_[idx]`, so there is no cross-frame contention.

### Tests

- The Python test `dali/test/python/operator_1/test_jpeg_compression_distortion.py`
  is the operator's existing coverage. It compares CPU and GPU outputs against
  each other and against reference fixtures. Expected to pass unchanged. If it
  fails purely on the comparison threshold vs. the GPU op (which is itself
  nvimgcodec/nvJPEG-based), bump the tolerance rather than loosen correctness.
- There is no C++ unit test for this operator today; none added.
- No unit test required for the new `NvImageCodecEncoder` wrapper itself; it's
  a thin handle wrapper transitively covered by the operator's tests.

### Risks

- **Performance regression.** Plugin dispatch overhead per `EncoderEncode` call
  vs. a direct libjpeg-turbo call. Should be measured once with the existing
  TL1 decoder-perf scaffolding before merging. The design already minimises
  call count by issuing one batched call per quality bucket; if a regression
  is still observed, the next escalation would be reusing one
  `nvimgcodecImage_t` / `nvimgcodecCodeStream_t` pool across `RunImpl` calls
  (out of scope for this change).
- **Runtime extension presence.** The operator now requires the
  `libjpeg_turbo_ext.so.0` extension to be present at runtime. The wheel's
  install-requires entry already pulls `nvidia-nvimgcodec-cu${CUDA}[all]` which
  includes the extension; manual installs that skip extensions will break.
- **Test goldens.** A version skew between OpenCV's libjpeg-turbo and
  nvimgcodec's may produce different pixel values for the same `quality`
  setting. Goldens are compared with tolerances; if a single test breaks,
  regenerate that golden, don't widen tolerance globally.
