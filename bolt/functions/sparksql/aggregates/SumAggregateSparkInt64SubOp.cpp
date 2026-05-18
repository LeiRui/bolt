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

#include "bolt/functions/sparksql/aggregates/SumAggregateSparkInt64SubOp.h"

#include "bolt/exec/AggregationHook.h"
#include "bolt/functions/lib/CheckedArithmeticImpl.h"
#include "bolt/vector/BaseVector.h"
#include "bolt/vector/DecodedVector.h"
#include "bolt/vector/LazyVector.h"
#include "bolt/vector/VectorEncoding.h"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

namespace bytedance::bolt::functions::aggregate::sparksql {

namespace {

static void sparkSumInt64UpdateSingle(int64_t& result, int64_t value) {
  if (::bytedance::bolt::functions::aggregate::Overflow) {
    result += value;
  } else {
    CHECK_ADD(result, value);
  }
}

#if defined(__aarch64__) && defined(__linux__)
// SVE is advertised on AT_HWCAP (HWCAP_SVE), not AT_HWCAP2 — see Linux
// arch/arm64/include/uapi/asm/hwcap.h.
#ifndef HWCAP_SVE
constexpr unsigned long kBoltHwcapSve = 1UL << 22;
#else
constexpr unsigned long kBoltHwcapSve = HWCAP_SVE;
#endif

static bool linuxAarch64RuntimeHasSve() {
  const unsigned long hwcap = getauxval(AT_HWCAP);
  return (hwcap & kBoltHwcapSve) != 0;
}
#else
static bool linuxAarch64RuntimeHasSve() {
  return false;
}
#endif

} // namespace

SumAggregateSparkInt64SubOp::SumAggregateSparkInt64SubOp(TypePtr resultType)
    : Base(std::move(resultType)) {}

void SumAggregateSparkInt64SubOp::addRawInput(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool mayPushdown) {
  const auto& arg = args[0];

  if (mayPushdown && arg->isLazy()) {
    Base::addRawInput(groups, rows, args, mayPushdown);
    return;
  }

  using ::bytedance::bolt::functions::aggregate::Overflow;
  if (this->numNulls_ && Overflow) {
    DecodedVector decoded(*arg, rows, !mayPushdown);
    const auto encoding = decoded.base()->encoding();
    if (mayPushdown && encoding == VectorEncoding::Simple::LAZY &&
        !arg->type()->isDecimal()) {
      bytedance::bolt::aggregate::SimpleCallableHook<
          int64_t,
          int64_t,
          void (*)(int64_t&, int64_t)>
          hook(
              offset_,
              nullByte_,
              nullMask_,
              groups,
              &numNulls_,
              sparkSumInt64UpdateSingle);
      auto indices = decoded.indices();
      decoded.base()->as<const LazyVector>()->load(
          ::bytedance::bolt::RowSet(indices, arg->size()), &hook);
      return;
    }
    if (decoded.mayHaveNulls()) {
#if defined(__aarch64__)
      if (linuxAarch64RuntimeHasSve() &&
          updateGroupsFromDecoded(groups, rows, decoded)) {
        return;
      }
#endif
    }
    Base::addRawInput(groups, rows, args, mayPushdown);
    return;
  }

  Base::addRawInput(groups, rows, args, mayPushdown);
}

void SumAggregateSparkInt64SubOp::addIntermediateResults(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool mayPushdown) {
  const auto& arg = args[0];

  if (mayPushdown && arg->isLazy()) {
    Base::addIntermediateResults(groups, rows, args, mayPushdown);
    return;
  }

  using ::bytedance::bolt::functions::aggregate::Overflow;
  if (this->numNulls_ && Overflow) {
    DecodedVector decoded(*arg, rows, !mayPushdown);
    const auto encoding = decoded.base()->encoding();
    if (mayPushdown && encoding == VectorEncoding::Simple::LAZY &&
        !arg->type()->isDecimal()) {
      bytedance::bolt::aggregate::SimpleCallableHook<
          int64_t,
          int64_t,
          void (*)(int64_t&, int64_t)>
          hook(
              offset_,
              nullByte_,
              nullMask_,
              groups,
              &numNulls_,
              sparkSumInt64UpdateSingle);
      auto indices = decoded.indices();
      decoded.base()->as<const LazyVector>()->load(
          ::bytedance::bolt::RowSet(indices, arg->size()), &hook);
      return;
    }
    if (decoded.mayHaveNulls()) {
#if defined(__aarch64__)
      if (linuxAarch64RuntimeHasSve() &&
          updateGroupsFromDecoded(groups, rows, decoded)) {
        return;
      }
#endif
    }
    Base::addIntermediateResults(groups, rows, args, mayPushdown);
    return;
  }

  Base::addIntermediateResults(groups, rows, args, mayPushdown);
}

#if !defined(__aarch64__)
bool SumAggregateSparkInt64SubOp::updateGroupsFromDecoded(
    char** /*groups*/,
    const SelectivityVector& /*rows*/,
    ::bytedance::bolt::DecodedVector& /*decoded*/) {
  return false;
}
#endif

} // namespace bytedance::bolt::functions::aggregate::sparksql
