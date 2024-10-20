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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/c_api.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"
#include "dali/test/tensor_test_utils.h"

using namespace std::string_literals;  // NOLINT(build/namespaces)

namespace dali {

namespace {

constexpr int batch_size = 12;
constexpr int num_thread = 4;
constexpr int device_id = 0;
constexpr int seed = 0;
constexpr bool pipelined = true;
constexpr int prefetch_queue_depth = 2;
constexpr bool async = true;
constexpr float output_size = 20.f;
constexpr cudaStream_t cuda_stream = 0;
const std::string input_name = "inputs"s;    // NOLINT
const std::string output_name = "outputs"s;  // NOLINT

template<typename Backend>
struct backend_to_device_type {
  static constexpr device_type_t value = CPU;
};

template<>
struct backend_to_device_type<GPUBackend> {
  static constexpr device_type_t value = GPU;
};

template<typename Backend>
struct the_other_backend {
  using type = GPUBackend;
};

template<>
struct the_other_backend<GPUBackend> {
  using type = CPUBackend;
};


template<typename Backend, device_type_t execution_device = backend_to_device_type<Backend>::value>
std::unique_ptr<Pipeline> GetTestPipeline(bool is_file_reader, const std::string &output_device) {
  auto pipe_ptr = std::make_unique<Pipeline>(batch_size, num_thread, device_id, seed, pipelined,
                                             prefetch_queue_depth, async);
  auto &pipe = *pipe_ptr;
  std::string exec_device = execution_device == CPU ? "cpu" : "gpu";
  TensorList<Backend> data;
  if (is_file_reader) {
    std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
    std::string file_list = file_root + "image_list.txt";
    pipe.AddOperator(OpSpec("FileReader")
                             .AddArg("device", "cpu")
                             .AddArg("file_root", file_root)
                             .AddArg("file_list", file_list)
                             .AddOutput("compressed_images", "cpu")
                             .AddOutput("labels", "cpu"));

    pipe.AddOperator(OpSpec("ImageDecoder")
                             .AddArg("device", "cpu")
                             .AddArg("output_type", DALI_RGB)
                             .AddInput("compressed_images", "cpu")
                             .AddOutput(input_name, "cpu"));
  } else {
    pipe.AddExternalInput(input_name);
  }
  //  Some Op
  pipe.AddOperator(OpSpec("Resize")
                           .AddArg("device", exec_device)
                           .AddArg("image_type", DALI_RGB)
                           .AddArg("resize_x", output_size)
                           .AddArg("resize_y", output_size)
                           .AddInput(input_name, exec_device)
                           .AddOutput(output_name, exec_device));

  std::vector<std::pair<std::string, std::string>> outputs = {{output_name, output_device}};

  pipe.SetOutputNames(outputs);
  return pipe_ptr;
}


// Takes Outptus from baseline and handle and compares them
// Allows only for uint8_t CPU/GPU output data to be compared
template <typename Backend>
void ComparePipelinesOutputs(daliPipelineHandle &handle, Pipeline &baseline,
                             unsigned int copy_output_flags = DALI_ext_default,
                             int batch_size = dali::batch_size) {
  dali::DeviceWorkspace ws;
  baseline.Outputs(&ws);
  daliOutput(&handle);

  EXPECT_EQ(daliGetNumOutput(&handle), ws.NumOutput());
  const int num_output = ws.NumOutput();
  for (int output = 0; output < num_output; output++) {
    EXPECT_EQ(daliNumTensors(&handle, output), batch_size);
    for (int elem = 0; elem < batch_size; elem++) {
      auto *shape = daliShapeAtSample(&handle, output, elem);
      auto ref_shape = ws.Output<Backend>(output).shape()[elem];
      int D = ref_shape.size();
      for (int d = 0; d < D; d++)
        EXPECT_EQ(shape[d], ref_shape[d]);
      EXPECT_EQ(shape[D], 0) << "Shapes in C API are 0-terminated";
      free(shape);
    }

    TensorList<CPUBackend> pipeline_output_cpu, c_api_output_cpu;
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    pipeline_output_cpu.Copy(ws.Output<Backend>(0), cuda_stream);

    TensorList<Backend> c_api_output;
    c_api_output.Resize(pipeline_output_cpu.shape(), TypeInfo::Create<uint8_t>());
    daliOutputCopy(&handle, c_api_output.raw_mutable_data(), 0,
                   backend_to_device_type<Backend>::value, 0, copy_output_flags);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    c_api_output_cpu.Copy(c_api_output, cuda_stream);
    CUDA_CALL(cudaDeviceSynchronize());
    Check(view<uint8_t>(pipeline_output_cpu), view<uint8_t>(c_api_output_cpu));
  }
}

}  // namespace

template<typename Backend>
class CApiTest : public ::testing::Test {
 protected:
  std::string output_device_ = backend_to_device_type<Backend>::value == CPU ? "cpu"s : "gpu"s;
};

using Backends = ::testing::Types<CPUBackend, GPUBackend>;
TYPED_TEST_SUITE(CApiTest, Backends);


TYPED_TEST(CApiTest, GetOutputNameTest) {
  std::string output0_name = "compressed_images";
  std::string output1_name = "labels";
  auto pipe_ptr = std::make_unique<Pipeline>(batch_size, num_thread, device_id, seed, pipelined,
                                             prefetch_queue_depth, async);
  auto &pipe = *pipe_ptr;
  std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
  std::string file_list = file_root + "image_list.txt";
  pipe.AddOperator(OpSpec("FileReader")
                       .AddArg("device", "cpu")
                       .AddArg("file_root", file_root)
                       .AddArg("file_list", file_list)
                       .AddOutput(output0_name, "cpu")
                       .AddOutput(output1_name, "cpu"));

  std::vector<std::pair<std::string, std::string>> outputs = {{output0_name, "cpu"},
                                                              {output1_name, "cpu"}};

  pipe.SetOutputNames(outputs);

  auto serialized = pipe.SerializeToProtobuf();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  ASSERT_EQ(daliGetNumOutput(&handle), 2);
  EXPECT_STREQ(daliGetOutputName(&handle, 0), output0_name.c_str());
  EXPECT_STREQ(daliGetOutputName(&handle, 1), output1_name.c_str());
}


TYPED_TEST(CApiTest, FileReaderPipe) {
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();
  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }

  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
}

TYPED_TEST(CApiTest, FileReaderDefaultPipe) {
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();
  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }

  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, serialized.c_str(), serialized.size());
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }

  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
}


TYPED_TEST(CApiTest, ExternalSourceSingleAllocPipe) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<CPUBackend> input_cpu;
  TensorList<TypeParam> input;
  input_cpu.Resize(input_shape, TypeInfo::Create<uint8_t>());
  auto pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input_cpu), 42 * i);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    input.Copy(input_cpu, cuda_stream);
    pipe_ptr->SetExternalInput(input_name, input);
    daliSetExternalInputBatchSize(&handle, input_name.c_str(), input_shape.num_samples());
    daliSetExternalInputAsync(&handle, input_name.c_str(), backend_to_device_type<TypeParam>::value,
                              input.raw_data(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                              input_shape.sample_dim(), nullptr, cuda_stream, DALI_ext_default);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }

  SequentialFill(view<uint8_t>(input_cpu), 42 * prefetch_queue_depth);
  // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
  input.Copy(input_cpu, cuda_stream);
  pipe_ptr->SetExternalInput(input_name, input);
  daliSetExternalInputAsync(&handle, input_name.c_str(), backend_to_device_type<TypeParam>::value,
                            input.raw_data(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                            input_shape.sample_dim(), "HWC", cuda_stream, DALI_ext_default);
  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
}


TYPED_TEST(CApiTest, ExternalSourceSingleAllocVariableBatchSizePipe) {
  TensorListShape<> reference_input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                             {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                             {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  int max_batch_size = reference_input_shape.num_samples();
  std::vector<TensorListShape<>> trimmed_input_shapes = {
      sample_range(reference_input_shape, 0, max_batch_size / 2),
      sample_range(reference_input_shape, 0, max_batch_size / 4),
      sample_range(reference_input_shape, 0, max_batch_size),
  };

  auto pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (auto &input_shape : trimmed_input_shapes) {
    pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
    pipe_ptr->Build();

    TensorList<CPUBackend> input_cpu;
    TensorList<TypeParam> input;
    input_cpu.Resize(input_shape, TypeInfo::Create<uint8_t>());

    for (int i = 0; i < prefetch_queue_depth; i++) {
      SequentialFill(view<uint8_t>(input_cpu), 42 * i);
      // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
      input.Copy(input_cpu, cuda_stream);
      pipe_ptr->SetExternalInput(input_name, input);
      daliSetExternalInputBatchSize(&handle, input_name.c_str(), input_shape.num_samples());
      daliSetExternalInputAsync(&handle, input_name.c_str(),
                                backend_to_device_type<TypeParam>::value, input.raw_data(),
                                dali_data_type_t::DALI_UINT8, input_shape.data(),
                                input_shape.sample_dim(), nullptr, cuda_stream, DALI_ext_default);
    }

    for (int i = 0; i < prefetch_queue_depth; i++) {
      pipe_ptr->RunCPU();
      pipe_ptr->RunGPU();
    }
    daliPrefetchUniform(&handle, prefetch_queue_depth);

    dali::DeviceWorkspace ws;
    for (int i = 0; i < prefetch_queue_depth; i++) {
      ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr, DALI_ext_default,
                                         input_shape.num_samples());
    }
  }
}


TYPED_TEST(CApiTest, ExternalSourceMultipleAllocPipe) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<CPUBackend> input_cpu;
  TensorList<TypeParam> input;
  input_cpu.Resize(input_shape, TypeInfo::Create<uint8_t>());
  std::vector<const void *> data_ptrs(batch_size);
  for (int i = 0; i < batch_size; i++) {
    data_ptrs[i] = input_cpu.raw_tensor(i);
  }
  auto pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input_cpu), 42 * i);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    input.Copy(input_cpu, cuda_stream);
    pipe_ptr->SetExternalInput(input_name, input, cuda_stream);
    daliSetExternalInputTensorsAsync(&handle, input_name.c_str(),
                                     backend_to_device_type<TypeParam>::value, data_ptrs.data(),
                                     dali_data_type_t::DALI_UINT8, input_shape.data(),
                                     input_shape.sample_dim(), nullptr, cuda_stream,
                                     DALI_ext_default);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }

  SequentialFill(view<uint8_t>(input_cpu), 42 * prefetch_queue_depth);
  // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
  input.Copy(input_cpu, cuda_stream);
  pipe_ptr->SetExternalInput(input_name, input, cuda_stream);
  daliSetExternalInputTensorsAsync(&handle, input_name.c_str(),
                                   backend_to_device_type<TypeParam>::value, data_ptrs.data(),
                                   dali_data_type_t::DALI_UINT8, input_shape.data(),
                                   input_shape.sample_dim(), "HWC", cuda_stream, DALI_ext_default);
  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
}


TYPED_TEST(CApiTest, ExternalSourceSingleAllocDifferentBackendsTest) {
  using OpBackend = TypeParam;
  using DataBackend = typename the_other_backend<TypeParam>::type;
  if (std::is_same<OpBackend, CPUBackend>::value && std::is_same<DataBackend, GPUBackend>::value) {
    GTEST_SKIP();  // GPU data -> CPU op   is currently not supported. Might be added later.
  }
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8,  8,  3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<CPUBackend> input_cpu;
  TensorList<DataBackend> input;
  input_cpu.Resize(input_shape, TypeInfo::Create<uint8_t>());
  auto pipe_ptr = GetTestPipeline<OpBackend>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input_cpu), 42 * i);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    input.Copy(input_cpu, cuda_stream);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    pipe_ptr->SetExternalInput(input_name, input);
    daliSetExternalInput(&handle, input_name.c_str(), backend_to_device_type<DataBackend>::value,
                         input.raw_data(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                         input_shape.sample_dim(), nullptr, DALI_ext_default);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<OpBackend>(handle, *pipe_ptr);
  }

  SequentialFill(view<uint8_t>(input_cpu), 42 * prefetch_queue_depth);
  // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
  input.Copy(input_cpu, cuda_stream);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
  pipe_ptr->SetExternalInput(input_name, input);
  daliSetExternalInput(&handle, input_name.c_str(), backend_to_device_type<DataBackend>::value,
                       input.raw_data(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                       input_shape.sample_dim(), "HWC", DALI_ext_default);
  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<OpBackend>(handle, *pipe_ptr);
}


TYPED_TEST(CApiTest, ExternalSourceMultipleAllocDifferentBackendsTest) {
  using OpBackend = TypeParam;
  using DataBackend = typename the_other_backend<TypeParam>::type;
  if (std::is_same<OpBackend, CPUBackend>::value && std::is_same<DataBackend, GPUBackend>::value) {
    GTEST_SKIP();  // GPU data -> CPU op   is currently not supported. Might be added later.
  }
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8,  8,  3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<CPUBackend> input_cpu;
  TensorList<DataBackend> input;
  input_cpu.Resize(input_shape, TypeInfo::Create<uint8_t>());
  std::vector<const void *> data_ptrs(batch_size);
  for (int i = 0; i < batch_size; i++) {
    data_ptrs[i] = input_cpu.raw_tensor(i);
  }
  auto pipe_ptr = GetTestPipeline<OpBackend>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input_cpu), 42 * i);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    input.Copy(input_cpu, cuda_stream);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    pipe_ptr->SetExternalInput(input_name, input, cuda_stream);
    daliSetExternalInputTensors(&handle, input_name.c_str(),
                                backend_to_device_type<DataBackend>::value, data_ptrs.data(),
                                dali_data_type_t::DALI_UINT8, input_shape.data(),
                                input_shape.sample_dim(), nullptr, DALI_ext_default);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<OpBackend>(handle, *pipe_ptr);
  }

  SequentialFill(view<uint8_t>(input_cpu), 42 * prefetch_queue_depth);
  // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
  input.Copy(input_cpu, cuda_stream);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
  pipe_ptr->SetExternalInput(input_name, input, cuda_stream);
  daliSetExternalInputTensors(&handle, input_name.c_str(),
                              backend_to_device_type<DataBackend>::value, data_ptrs.data(),
                              dali_data_type_t::DALI_UINT8, input_shape.data(),
                              input_shape.sample_dim(), "HWC", DALI_ext_default);
  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<OpBackend>(handle, *pipe_ptr);
}

TYPED_TEST(CApiTest, TestExecutorMeta) {
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr.reset();
  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, true);

  daliRun(&handle);
  daliOutput(&handle);
  CUDA_CALL(cudaDeviceSynchronize());

  size_t N;
  daliExecutorMetadata *meta;
  daliGetExecutorMetadata(&handle, &meta, &N);
  EXPECT_EQ(N, 4);
  for (size_t i = 0; i< N; ++i) {
    auto &meta_entry = meta[i];
    for (size_t j = 0; j < meta_entry.out_num; ++j) {
      EXPECT_LE(meta_entry.real_size[j], meta_entry.reserved[j]);
    }
  }
  daliFreeExecutorMetadata(meta, N);
}

TYPED_TEST(CApiTest, UseCopyKernel) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<CPUBackend> input_cpu;
  input_cpu.Resize(input_shape, TypeInfo::Create<uint8_t>());

  TensorList<TypeParam> input;
  if (std::is_same<TypeParam, CPUBackend>::value)
    input.set_pinned(true);

  auto pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  unsigned int flags = DALI_ext_default | DALI_ext_force_sync | DALI_use_copy_kernel;
  if (std::is_same<TypeParam, CPUBackend>::value)
    flags |= DALI_ext_pinned;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input_cpu), 42 * i);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    input.Copy(input_cpu, cuda_stream);
    pipe_ptr->SetExternalInput(input_name, input);
    daliSetExternalInputAsync(&handle, input_name.c_str(), backend_to_device_type<TypeParam>::value,
                              input.raw_data(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                              input_shape.sample_dim(), nullptr, cuda_stream, flags);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr, flags);
  }
}

template <typename Backend>
void Clear(Tensor<Backend>& tensor);

template <>
void Clear(Tensor<CPUBackend>& tensor) {
  std::memset(tensor.raw_mutable_data(), 0, tensor.nbytes());
}

template <>
void Clear(Tensor<GPUBackend>& tensor) {
  CUDA_CALL(cudaMemset(tensor.raw_mutable_data(), 0, tensor.nbytes()));
}


TYPED_TEST(CApiTest, daliOutputCopySamples) {
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, serialized.c_str(), serialized.size());
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  daliRun(&handle);
  daliOutput(&handle);
  const int num_output = daliGetNumOutput(&handle);
  for (int out_idx = 0; out_idx < num_output; out_idx++) {
    std::vector<int64_t> sample_sizes(batch_size, 0);
    EXPECT_EQ(daliNumTensors(&handle, out_idx), batch_size);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto *shape = daliShapeAtSample(&handle, out_idx, sample_idx);
      int ndim = 0;
      sample_sizes[sample_idx] = 1;
      for (int d = 0; shape[d] > 0; d++) {
        sample_sizes[sample_idx] *= shape[d];
      }
      free(shape);
    }

    DALIDataType type = static_cast<DALIDataType>(daliTypeAt(&handle, out_idx));
    auto type_info = dali::TypeTable::GetTypeInfo(type);
    int64_t out_size = daliNumElements(&handle, out_idx);
    Tensor<TypeParam> output1;
    output1.Resize({out_size}, type_info);
    daliOutputCopy(&handle, output1.raw_mutable_data(), out_idx,
                   backend_to_device_type<TypeParam>::value, 0, DALI_ext_default);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    Tensor<CPUBackend> output1_cpu;
    output1_cpu.Copy(output1, cuda_stream);

    for (bool use_copy_kernel : {false, true}) {
      Tensor<TypeParam> output2;
      Tensor<CPUBackend> output2_cpu;
      output2.set_pinned(std::is_same<TypeParam, CPUBackend>::value);
      output2.Resize({out_size}, type_info);
      // Making sure data is cleared
      // Somehow in debug mode it can get the same raw pointer which happen to have
      // the right data in the second iteration
      Clear(output2);

      std::vector<void*> sample_dsts(batch_size);
      int64_t offset = 0;
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        sample_dsts[sample_idx] = static_cast<uint8_t*>(output2.raw_mutable_data()) + offset;
        offset += sample_sizes[sample_idx] * type_info.size();
      }

      unsigned int flags = DALI_ext_default;
      if (use_copy_kernel)
        flags |= DALI_use_copy_kernel;
      if (std::is_same<TypeParam, CPUBackend>::value)
        flags |= DALI_ext_pinned;

      daliOutputCopySamples(&handle, sample_dsts.data(), out_idx,
                            backend_to_device_type<TypeParam>::value, cuda_stream, flags);

      // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
      output2_cpu.Copy(output2, cuda_stream);
      CUDA_CALL(cudaDeviceSynchronize());
      Check(view<uint8_t>(output1_cpu), view<uint8_t>(output2_cpu));
    }

    for (bool use_copy_kernel : {false, true}) {
      Tensor<TypeParam> output2;
      Tensor<CPUBackend> output2_cpu;
      output2.set_pinned(std::is_same<TypeParam, CPUBackend>::value);
      output2.Resize({out_size}, type_info);
      // Making sure data is cleared
      // Somehow in debug mode it can get the same raw pointer which happen to have
      // the right data in the second iteration
      Clear(output2);

      std::vector<void*> sample_dsts_even(batch_size);
      std::vector<void*> sample_dsts_odd(batch_size);
      int64_t offset = 0;
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        auto sample_ptr = static_cast<uint8_t*>(output2.raw_mutable_data()) + offset;
        if (sample_idx % 2 == 0) {
          sample_dsts_even[sample_idx] = sample_ptr;
          sample_dsts_odd[sample_idx]  = nullptr;
        } else {
          sample_dsts_even[sample_idx] = nullptr;
          sample_dsts_odd[sample_idx]  = sample_ptr;
        }
        offset += sample_sizes[sample_idx] * type_info.size();
      }

      unsigned int flags = DALI_ext_default;
      if (use_copy_kernel)
        flags |= DALI_use_copy_kernel;
      if (std::is_same<TypeParam, CPUBackend>::value)
        flags |= DALI_ext_pinned;

      daliOutputCopySamples(&handle, sample_dsts_even.data(), out_idx,
                            backend_to_device_type<TypeParam>::value, cuda_stream, flags);
      daliOutputCopySamples(&handle, sample_dsts_odd.data(), out_idx,
                            backend_to_device_type<TypeParam>::value, cuda_stream, flags);

      // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
      output2_cpu.Copy(output2, cuda_stream);
      CUDA_CALL(cudaDeviceSynchronize());
      Check(view<uint8_t>(output1_cpu), view<uint8_t>(output2_cpu));
    }
  }
}

TYPED_TEST(CApiTest, CpuOnlyTest) {
  dali::Pipeline pipe(1, 1, dali::CPU_ONLY_DEVICE_ID);
  pipe.AddExternalInput("dummy");
  std::vector<std::pair<std::string, std::string>> outputs = {{"dummy", "cpu"}};
  pipe.SetOutputNames(outputs);
  std::string ser = pipe.SerializeToProtobuf();
  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, ser.c_str(), ser.size());
}

}  // namespace dali
