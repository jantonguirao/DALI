
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/common/find/find_first_last_gpu.cuh"
#include "dali/pipeline/data/views.h"

namespace dali {

namespace {

template <typename T>
struct threshold_ptr {
  T* ptr_;
  DALI_HOST_DEV DALI_FORCEINLINE bool operator()(T x) const noexcept {
    return x >= *ptr_;
  }
};

template <typename T>
struct threshold_val {
  T value_;
  explicit threshold_val(T value = 0) : value_(value) {}

  DALI_HOST_DEV DALI_FORCEINLINE bool operator()(T x) const noexcept {
    return x >= value_;
  }
};

}  // namespace

class NonsilenceOperatorGpu : public NonsilenceOperator<GPUBackend> {
 public:
  explicit NonsilenceOperatorGpu(const OpSpec &spec) :
          NonsilenceOperator<GPUBackend>(spec) {}

  ~NonsilenceOperatorGpu() override = default;
  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperatorGpu);

 protected:
  void RunImpl(workspace_t<GPUBackend> &ws) override {
    auto dtype = ws.template Input<GPUBackend>(0).type();
    // TYPE_SWITCH(dtype, type2id, T, NONSILENCE_TYPES, (
      RunImplTyped<float>(ws);
   // ), DALI_FAIL(make_string("Unsupported input type: ", dtype));)  // NOLINT
  }

 private:
  kernels::MaxGPU<float, float> max_kernel;
  kernels::find::FindFirstLastGPU find_first_last_kernel;

  template <typename T>
  void CalcMMS(kernels::KernelContext &ctx, TensorListView<StorageGPU, float, 1> &mms,
               const TensorListView<StorageGPU, const T, 1> &in) {
    kernels::signal::MovingMeanSquareGpu<T> kernel;
    kernels::signal::MovingMeanSquareArgs args{window_length_, reset_interval_};
    kernel.Run(ctx, mms, in, args);
  }

  void CalcMax(kernels::KernelContext &ctx, TensorListView<StorageGPU, float, 0> &max,
               const TensorListView<StorageGPU, float, 1> &in) {
    std::array<int, 1> axes = { 0 };
    max_kernel.Setup(ctx, in.shape, make_cspan(axes), false, false);
    max_kernel.Run(ctx, max, in);
  }

  void CalcNonsilentRegionRefMax(kernels::KernelContext &ctx,
                                 TensorListView<StorageGPU, int32_t, 0> &begin,
                                 TensorListView<StorageGPU, int32_t, 0> &len,
                                 TensorListView<StorageGPU, float, 1> &mms) {
    int nsamples = mms.num_samples();
    TensorListShape<0> scalar(nsamples);
    auto max_mms = ctx.scratchpad->AllocTensorList<mm::memory_kind::device, float, 0>(scalar);
    CalcMax(ctx, max_mms, mms);

    using Predicate = threshold_ptr<float>;
    // no need for it to be pinned, since we copy those predicates to sample descriptors (pinned) in the kernel
    span<Predicate> predicates{ctx.scratchpad->Allocate<mm::memory_kind::host, Predicate>(nsamples), nsamples};
    for (int i = 0; i < nsamples; i++) {
      predicates[i].ptr_ = max_mms[i].data;
    }

    using OutFormat = kernels::find::begin_length<int32_t>;
    find_first_last_kernel.template Run<float, int32_t, Predicate, OutFormat>(
        ctx, begin, len, mms, predicates);
  }

  void CalcNonsilentRegionRefConstant(kernels::KernelContext &ctx,
                                      TensorListView<StorageGPU, int32_t, 0> &begin,
                                      TensorListView<StorageGPU, int32_t, 0> &len,
                                      TensorListView<StorageGPU, float, 1> &mms) {
    int nsamples = mms.num_samples();
    TensorListShape<0> scalar(nsamples);
    auto max_mms = ctx.scratchpad->AllocTensorList<mm::memory_kind::device, float, 0>(scalar);
    CalcMax(ctx, max_mms, mms);

    using Predicate = threshold_ptr<float>;
    // no need for it to be pinned, since we copy those predicates to sample descriptors (pinned) in
    // the kernel
    span<Predicate> predicates{ctx.scratchpad->Allocate<mm::memory_kind::host, Predicate>(nsamples),
                               nsamples};
    for (int i = 0; i < nsamples; i++) {
      predicates[i].ptr_ = max_mms[i].data;
    }

    using OutFormat = kernels::find::begin_length<int32_t>;
    find_first_last_kernel.template Run<float, int32_t, Predicate, OutFormat>(ctx, begin, len, mms,
                                                                              predicates);
  }


  template <typename T>
  void RunImplTyped(workspace_t<GPUBackend> &ws) {
    kernels::DynamicScratchpad scratchpad({}, ws.stream());
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    ctx.scratchpad = &scratchpad;

    auto input = view<const T, 1>(ws.template Input<GPUBackend>(0));
    int nsamples = input.shape.num_samples();
    auto out_begin = view<int32_t>(ws.template Output<GPUBackend>(0));
    auto out_end = view<int32_t>(ws.template Output<GPUBackend>(1));

    auto mms = scratchpad.AllocTensorList<mm::memory_kind::device, float, 1>(input.shape);
    CalcMMS(ctx, mms, input);

    if (reference_max_) {
      TensorListShape<0> scalar(nsamples);
      auto max_mms = scratchpad.AllocTensorList<mm::memory_kind::device, float, 0>(scalar);
      CalcMax(ctx, max_mms, mms);
    }
  }
};

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorGpu, GPU);


}  // namespace dali