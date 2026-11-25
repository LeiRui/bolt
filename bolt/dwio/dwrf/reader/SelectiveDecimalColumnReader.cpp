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

#include "bolt/dwio/dwrf/reader/SelectiveDecimalColumnReader.h"
namespace bytedance::bolt::dwrf {

template <typename DataT>
SelectiveDecimalColumnReader<DataT>::SelectiveDecimalColumnReader(
    const std::shared_ptr<const TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec)
    : SelectiveColumnReader(fileType->type(), fileType, params, scanSpec) {
  EncodingKey encodingKey{fileType_->id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  if constexpr (std::is_same_v<DataT, std::int64_t>) {
    scale_ = requestedType_->asShortDecimal().scale();
  } else {
    scale_ = requestedType_->asLongDecimal().scale();
  }
  version_ = convertRleVersion(stripe.getEncoding(encodingKey).kind());
  auto data = encodingKey.forKind(proto::Stream_Kind_DATA);
  valueUsesVarints_ = stripe.getUseVInts(data);
  valueDecoder_ = createDirectDecoder</*isSigned*/ true>(
      stripe.getStream(data, params.streamLabels().label(), true),
      valueUsesVarints_,
      sizeof(DataT));

  // [NOTICE] DWRF's NANO_DATA has the same enum value as ORC's SECONDARY
  auto secondary = encodingKey.forKind(proto::Stream_Kind_NANO_DATA);
  scaleDecoder_ = createRleDecoder</*isSigned*/ true>(
      stripe.getStream(secondary, params.streamLabels().label(), true),
      version_,
      memoryPool_,
      stripe.getUseVInts(secondary),
      LONG_BYTE_SIZE);
}

template <typename DataT>
uint64_t SelectiveDecimalColumnReader<DataT>::skip(uint64_t numValues) {
  numValues = SelectiveColumnReader::skip(numValues);
  valueDecoder_->skip(numValues);
  scaleDecoder_->skip(numValues);
  return numValues;
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::seekToRowGroup(int64_t index) {
  SelectiveColumnReader::seekToRowGroup(index);
  auto positionsProvider = formatData_->seekToRowGroup(index);
  valueDecoder_->seekToRowGroup(positionsProvider);
  scaleDecoder_->seekToRowGroup(positionsProvider);
  BOLT_CHECK(!positionsProvider.hasNext());
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::processNulls(
    const bool isNull,
    const RowSet rows,
    const uint64_t* rawNulls) {
  if (!rawNulls) {
    return;
  }

  auto rawDecimal = values_->asMutable<DataT>();
  auto rawScale = scaleBuffer_->asMutable<int64_t>();

  returnReaderNulls_ = false;
  anyNulls_ = !isNull;
  allNull_ = isNull;

  vector_size_t idx = 0;
  for (vector_size_t i = 0; i < numValues_; i++) {
    if (isNull) {
      if (bits::isBitNull(rawNulls, i)) {
        bits::setNull(rawResultNulls_, idx);
        addOutputRow(rows[i]);
        idx++;
      }
    } else {
      if (!bits::isBitNull(rawNulls, i)) {
        bits::setNull(rawResultNulls_, idx, false);
        rawDecimal[idx] = rawDecimal[i];
        rawScale[idx] = rawScale[i];
        addOutputRow(rows[i]);
        idx++;
      }
    }
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::processFilter(
    const RowSet rows,
    const uint64_t* rawNulls) {
  auto filter = scanSpec_->filter();
  BOLT_CHECK_NOT_NULL(filter);
  auto rawDecimal = values_->asMutable<DataT>();

  returnReaderNulls_ = false;
  anyNulls_ = false;
  allNull_ = true;

  vector_size_t idx = 0;

  for (vector_size_t i = 0; i < numValues_; i++) {
    if (rawNulls && bits::isBitNull(rawNulls, i)) {
      if (filter->testNull()) {
        bits::setNull(rawResultNulls_, idx);
        addOutputRow(rows[i]);
        anyNulls_ = true;
        idx++;
      }
    } else {
      bool testRes = false;
      if constexpr (std::is_same_v<DataT, std::int64_t>) {
        testRes = filter->testInt64(rawDecimal[i]);
      } else {
        testRes = filter->testInt128(rawDecimal[i]);
      }
      if (testRes) {
        if (rawNulls) {
          bits::setNull(rawResultNulls_, idx, false);
        }
        rawDecimal[idx] = rawDecimal[i];
        addOutputRow(rows[i]);
        allNull_ = false;
        idx++;
      }
    }
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::process(
    // const common::Filter* filter,
    const RowSet& rows,
    const uint64_t* rawNulls) {
  auto filter = scanSpec_->filter();
  if (!filter) {
    // No filter and "hasDeletion" is false so input rows will be
    // reused.
    return;
  }

  switch ((filter->kind() == common::FilterKind::kIsNotNull && !rawNulls)
              ? common::FilterKind::kAlwaysTrue
              : filter->kind()) {
    case common::FilterKind::kAlwaysTrue:
      for (vector_size_t i = 0; i < numValues_; i++) {
        addOutputRow(rows[i]);
      }
      break;
    case common::FilterKind::kIsNull:
      processNulls(true, rows, rawNulls);
      break;
    case common::FilterKind::kIsNotNull:
      BOLT_CHECK(rawNulls);
      processNulls(false, rows, rawNulls);
      break;
    default:
      processFilter(rows, rawNulls);
  }
}

template <typename DataT>
template <bool kDense>
void SelectiveDecimalColumnReader<DataT>::readHelper(const RowSet& rows) {
  vector_size_t numRows = rows.back() + 1;
  ExtractToReader extractValues(this);
  common::AlwaysTrue alwaysTrue;
  DirectRleColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      decltype(extractValues),
      kDense>
      visitor(alwaysTrue, this, rows, extractValues);

  // decode scale stream
  if (version_ == bolt::dwrf::RleVersion_1) {
    decodeWithVisitor<RleDecoderV1<true>>(scaleDecoder_.get(), visitor);
  } else {
    decodeWithVisitor<RleDecoderV2<true>>(scaleDecoder_.get(), visitor);
  }

  // copy scales into scaleBuffer_
  ensureCapacity<int64_t>(scaleBuffer_, numValues_, &memoryPool_);
  scaleBuffer_->setSize(numValues_ * sizeof(int64_t));
  memcpy(
      scaleBuffer_->asMutable<char>(),
      rawValues_,
      numValues_ * sizeof(int64_t));

  // reset numValues_ before reading values
  numValues_ = 0;
  valueSize_ = sizeof(DataT);
  ensureValuesCapacity<DataT>(numRows);

  // decode value stream
  bytedance::bolt::dwio::common::
      ColumnVisitor<DataT, common::AlwaysTrue, decltype(extractValues), kDense>
          valueVisitor(alwaysTrue, this, rows, extractValues);
#if defined(__aarch64__)
  if constexpr (std::is_same_v<DataT, std::int64_t>) {
    decodeWithVisitor<DirectDecoder<true>>(
        valueDecoder_.get(), valueVisitor, !valueUsesVarints_);
  } else
#endif
  {
    decodeWithVisitor<DirectDecoder<true>>(valueDecoder_.get(), valueVisitor);
  }
  readOffset_ += numRows;

  fillDecimals();

  const auto rawNulls = nullsInReadRange_
      ? (kDense ? nullsInReadRange_->as<uint64_t>() : rawResultNulls_)
      : nullptr;

  process(rows, rawNulls);
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  BOLT_CHECK(!scanSpec_->valueHook());
  prepareRead<int64_t>(offset, rows, incomingNulls);
  if (!scanSpec_->keepValues() && scanSpec_->filter() &&
      (!resultNulls_ || !resultNulls_->unique() ||
       resultNulls_->capacity() * 8 < rows.size())) {
    // Make sure a dedicated resultNulls_ is allocated with enough capacity as
    // RleDecoder always assumes it is available.
    resultNulls_ = AlignedBuffer::allocate<bool>(rows.size(), &memoryPool_);
    rawResultNulls_ = resultNulls_->asMutable<uint64_t>();
  }
  const bool isDense = rows.back() == rows.size() - 1;
  if (isDense) {
    readHelper<true>(rows);
  } else {
    readHelper<false>(rows);
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::getValues(
    const RowSet& rows,
    VectorPtr* result) {
  auto nullsPtr =
      resultNulls() ? resultNulls()->template as<uint64_t>() : nullptr;
  auto scales = scaleBuffer_->as<int64_t>();
  auto values = values_->asMutable<DataT>();
  rawValues_ = values_->asMutable<char>();
  getIntValues(rows, requestedType_, result);
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::fillDecimals() {
  auto nullsPtr =
      resultNulls() ? resultNulls()->template as<uint64_t>() : nullptr;
  auto scales = scaleBuffer_->as<int64_t>();
  auto values = values_->asMutable<DataT>();

  DecimalUtil::fillDecimals<DataT>(
      values, nullsPtr, values, scales, numValues_, scale_);
}

template class SelectiveDecimalColumnReader<int64_t>;
template class SelectiveDecimalColumnReader<int128_t>;

} // namespace bytedance::bolt::dwrf
