// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dali_test_matching.h"

namespace dali {

template <typename ImgType>
class ElementExtractTest : public GenericMatchingTest<ImgType> {
 protected:
  template <typename T, int C = 3>
  void PrepareInput(TensorList<CPUBackend>& data, int ntensors = 2) {
    std::vector<Dims> shape;
    for (int i=0; i<ntensors; i++) {
        shape.push_back({10, 1280, 720, 3});
    }
    data.set_type(TypeInfo::Create<T>());
    data.SetLayout(DALITensorLayout::DALI_NFHWC);
    data.Resize(shape);

    // TODO(janton) fill this in with data
  }

  virtual uint32_t GetTestCheckType() const override {
    return t_checkColorComp;  // + t_checkAll + t_checkNoAssert;
  }

  void RunTestImpl(const opDescr &descr) override {
    const int batch_size = 2;
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> data;
    PrepareInput<float>(data);
    this->SetExternalInputs({{"input", &data}});

    // Launching the same transformation on CPU (outputIdx 0) and GPU (outputIdx 1)
    this->AddOperatorWithOutput(descr);
    this->RunOperator(descr);

    // TODO(janton) remove this
    pipe->SaveGraphToDotFile("graph.dot");
  }

};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(ElementExtractTest, Types);


TYPED_TEST(ElementExtractTest, Test1) {
    this->RunTest( {"ElementExtract", {"element_map", "1,2,3", DALI_INT_VEC}, 0.0});
}

}  // namespace dali
