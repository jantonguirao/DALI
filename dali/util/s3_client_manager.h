// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_S3_CLIENT_MANAGER_H_
#define DALI_UTIL_S3_CLIENT_MANAGER_H_

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <cstdio>
#include <memory>
#include <string>
#include "dali/core/common.h"

namespace dali {

struct S3ClientManager {
 public:
  static S3ClientManager& Instance() {
    static S3ClientManager s_manager_;
    return s_manager_;
  }

  Aws::S3::S3Client* client() {
    return client_.get();
  }

 private:
  S3ClientManager() {
    Aws::InitAPI(options_);
    // Aws::Client::ClientConfiguration clientConfig;
    // TODO(janton): fix this when the env vars are not present
    // clientConfig.region = std::getenv("DALI_AWS_REGION");
    // clientConfig.endpointOverride = std::getenv("DALI_AWS_ENDPOINT");
    client_ = std::make_unique<Aws::S3::S3Client>();
  }

  ~S3ClientManager() {
    Aws::ShutdownAPI(options_);
  }

  Aws::SDKOptions options_;
  std::unique_ptr<Aws::S3::S3Client> client_;
};

}  // namespace dali

#endif  // DALI_UTIL_S3_CLIENT_MANAGER_H_
