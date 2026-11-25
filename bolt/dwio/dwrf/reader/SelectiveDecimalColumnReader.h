/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 * Copyright (c) ByteDance Ltd. and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * This file has been modified by ByteDance Ltd. and/or its affiliates on
 * 2025-11-11.
 *
 * Original file was released under the Apache License 2.0,
 * with the full license text available at:
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This modified file is released under the same license.
 * --------------------------------------------------------------------------
 */

#pragma once

#include "bolt/dwio/common/BufferUtil.h"
#include "bolt/dwio/common/SelectiveColumnReaderInternal.h"
#include "bolt/dwio/dwrf/common/DecoderUtil.h"
#include "bolt/dwio/dwrf/reader/DwrfData.h"
namespace bytedance::bolt::dwrf {

using namespace dwio::common;

template <typename DataT>
class SelectiveDecimalColumnReader : public SelectiveColumnReader {
 public:
  SelectiveDecimalColumnReader(
      const std::shared_ptr<const TypeWithId>& fileType,
      DwrfParams& params,
      common::ScanSpec& scanSpec);

  void seekToRowGroup(int64_t index) override;
  uint64_t skip(uint64_t numValues) override;

  void read(int64_t offset, const RowSet& rows, const uint64_t* nulls) override;

  void getValues(const RowSet& rows, VectorPtr* result) override;

  bool hasBulkPath() const override {
    return false;
  }

 private:
  template <bool kDense>
  void readHelper(const RowSet& rows);

  void
  processNulls(const bool isNull, const RowSet rows, const uint64_t* rawNulls);

  void processFilter(const RowSet rows, const uint64_t* rawNulls);

  void process(const RowSet& rows, const uint64_t* rawNulls);

  void fillDecimals();

  std::unique_ptr<IntDecoder<true>> valueDecoder_;
  std::unique_ptr<IntDecoder<true>> scaleDecoder_;

  BufferPtr scaleBuffer_;
  RleVersion version_;
  int32_t scale_ = 0;
  bool valueUsesVarints_;
};

} // namespace bytedance::bolt::dwrf
