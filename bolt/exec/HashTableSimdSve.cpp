/*
 * Copyright (c) ByteDance Ltd. and/or its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "bolt/exec/HashTableSimd.h"

#include <arm_sve.h>

namespace bytedance::bolt::exec::hash_table_simd {
namespace {

// Exactly four u64 lanes, including when the hardware vector is wider.
// 256-bit SVE fills the register; the caller only enters this file when
// svcntb()==32.
svbool_t pgU64x4() {
  return svwhilelt_b64(uint64_t{0}, uint64_t{4});
}

} // namespace

namespace {

uint32_t stopBits4(
    const uint64_t* slots,
    uint64_t tag,
    uint32_t laneBegin,
    uint32_t laneEnd,
    uint32_t bitBase) {
  const svbool_t pg = pgU64x4();
  const svuint64_t values = svld1_u64(pg, slots);
  const svuint64_t tags = svand_n_u64_x(pg, values, kTagMask);
  const svbool_t tagHit = svcmpeq_n_u64(pg, tags, tag);
  const svbool_t empty = svcmpeq_n_u64(pg, values, 0);
  svbool_t stop = svorr_b_z(pg, tagHit, empty);

  const svuint64_t index = svindex_u64(bitBase, 1);
  const svbool_t inRange = svand_b_z(
      pg,
      svcmpge_n_u64(pg, index, laneBegin),
      svcmplt_n_u64(pg, index, laneEnd));
  stop = svand_b_z(pg, stop, inRange);

  uint64_t flags[4] = {};
  svst1_u64(pg, flags, svsel_u64(stop, svdup_n_u64(1), svdup_n_u64(0)));
  uint32_t bits = 0;
  for (int lane = 0; lane < 4; ++lane) {
    bits |= static_cast<uint32_t>(flags[lane]) << (bitBase + lane);
  }
  return bits;
}

} // namespace

uint32_t sveBucketStopMask(
    char* const* table,
    uint64_t bucket,
    uint64_t tag,
    uint32_t laneBegin,
    uint32_t laneEnd) {
  const auto* slots = reinterpret_cast<const uint64_t*>(table + bucket);
  // Both halves stay unconditional, matching scanBucketChain on AVX2.
  return stopBits4(slots, tag, laneBegin, laneEnd, 0) |
      stopBits4(slots + 4, tag, laneBegin, laneEnd, 4);
}

void sveAdvanceSlots(uint64_t* activeSlots, int numSlots, uint64_t mask) {
  const svbool_t pg = pgU64x4();
  int i = 0;
  for (; i + 4 <= numSlots; i += 4) {
    svuint64_t values = svld1_u64(pg, activeSlots + i);
    values = svand_n_u64_x(pg, svadd_n_u64_x(pg, values, 1), mask);
    svst1_u64(pg, activeSlots + i, values);
  }
  for (; i < numSlots; ++i) {
    activeSlots[i] = (activeSlots[i] + 1) & mask;
  }
}

} // namespace bytedance::bolt::exec::hash_table_simd
