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

#include "bolt/common/base/BitUtil.h"
#include "bolt/common/base/Nulls.h"
#include "bolt/exec/VectorHasher.h"

#include <folly/Bits.h>

#include <arm_sve.h>

#include <memory>
#include <type_traits>

namespace bytedance::bolt::exec::hash_table_simd {
namespace {

constexpr int32_t kMinSelectivityDenominator = 5;

svbool_t pgU64x4() {
  return svwhilelt_b64(uint64_t{0}, uint64_t{4});
}

svbool_t pgU32x8() {
  return svwhilelt_b32(uint32_t{0}, uint32_t{8});
}

// Same steps as twangMix64Avx2 / folly::hasher<uint64_t>.
void twangMix64x4(uint64_t* values) {
  const svbool_t pg = pgU64x4();
  svuint64_t key = svld1_u64(pg, values);
  // (~key) + (key << 21). XOR with all-ones is the bitwise not.
  const svuint64_t ones = svdup_n_u64(~uint64_t{0});
  key = svadd_u64_x(
      pg, sveor_u64_x(pg, key, ones), svlsl_n_u64_x(pg, key, 21));
  key = sveor_u64_x(pg, key, svlsr_n_u64_x(pg, key, 24));
  key = svadd_u64_x(
      pg,
      svadd_u64_x(pg, key, svlsl_n_u64_x(pg, key, 3)),
      svlsl_n_u64_x(pg, key, 8));
  key = sveor_u64_x(pg, key, svlsr_n_u64_x(pg, key, 14));
  key = svadd_u64_x(
      pg,
      svadd_u64_x(pg, key, svlsl_n_u64_x(pg, key, 2)),
      svlsl_n_u64_x(pg, key, 4));
  key = sveor_u64_x(pg, key, svlsr_n_u64_x(pg, key, 28));
  key = svadd_u64_x(pg, key, svlsl_n_u64_x(pg, key, 31));
  svst1_u64(pg, values, key);
}

// Same steps as jenkinsRevMix32Avx2 / folly::hash::jenkins_rev_mix32.
void jenkinsRevMix32x8(uint32_t* values) {
  const svbool_t pg = pgU32x8();
  svuint32_t key = svld1_u32(pg, values);
  key = svadd_u32_x(pg, key, svlsl_n_u32_x(pg, key, 12));
  key = sveor_u32_x(pg, key, svlsr_n_u32_x(pg, key, 22));
  key = svadd_u32_x(pg, key, svlsl_n_u32_x(pg, key, 4));
  key = sveor_u32_x(pg, key, svlsr_n_u32_x(pg, key, 9));
  key = svadd_u32_x(pg, key, svlsl_n_u32_x(pg, key, 10));
  key = sveor_u32_x(pg, key, svlsr_n_u32_x(pg, key, 2));
  key = svadd_u32_x(pg, key, svlsl_n_u32_x(pg, key, 7));
  key = svadd_u32_x(pg, key, svlsl_n_u32_x(pg, key, 12));
  svst1_u32(pg, values, key);
}

// bits::hashMix, four lanes. `upper` is read and overwritten.
void hashMix64x4(uint64_t* upper, const uint64_t* lower) {
  const svbool_t pg = pgU64x4();
  const svuint64_t kMul = svdup_n_u64(0x9ddfea08eb382d69ULL);
  const svuint64_t up = svld1_u64(pg, upper);
  const svuint64_t lo = svld1_u64(pg, lower);
  svuint64_t a = svmul_u64_x(pg, sveor_u64_x(pg, lo, up), kMul);
  a = sveor_u64_x(pg, a, svlsr_n_u64_x(pg, a, 47));
  svuint64_t b = svmul_u64_x(pg, sveor_u64_x(pg, up, a), kMul);
  b = sveor_u64_x(pg, b, svlsr_n_u64_x(pg, b, 47));
  b = svmul_u64_x(pg, b, kMul);
  svst1_u64(pg, upper, b);
}

void applyNulls4(
    uint64_t* hashed,
    const uint64_t* nullBits,
    int32_t row) {
  for (int lane = 0; lane < 4; ++lane) {
    if (bits::isBitNull(nullBits, row + lane)) {
      hashed[lane] = BaseVector::kNullHash;
    }
  }
}

template <bool kMix>
void storeHashes4(uint64_t* output, uint64_t* hashed) {
  if constexpr (kMix) {
    hashMix64x4(output, hashed);
  } else {
    const svbool_t pg = pgU64x4();
    svst1_u64(pg, output, svld1_u64(pg, hashed));
  }
}

template <bool kMix, bool kHasNulls>
void hashDenseInt64(
    const int64_t* values,
    const uint64_t* nullBits,
    int32_t begin,
    int32_t end,
    uint64_t* output) {
  constexpr int kLanes = 4;
  int32_t row = begin;
  for (; row + kLanes <= end; row += kLanes) {
    uint64_t hashed[4];
    for (int lane = 0; lane < kLanes; ++lane) {
      hashed[lane] = static_cast<uint64_t>(values[row + lane]);
    }
    twangMix64x4(hashed);
    if constexpr (kHasNulls) {
      applyNulls4(hashed, nullBits, row);
    }
    storeHashes4<kMix>(output + row, hashed);
  }
  for (; row < end; ++row) {
    if constexpr (kHasNulls) {
      if (bits::isBitNull(nullBits, row)) {
        if constexpr (kMix) {
          output[row] = bits::hashMix(output[row], BaseVector::kNullHash);
        } else {
          output[row] = BaseVector::kNullHash;
        }
        continue;
      }
    }
    const uint64_t hashValue = folly::hasher<int64_t>()(values[row]);
    if constexpr (kMix) {
      output[row] = bits::hashMix(output[row], hashValue);
    } else {
      output[row] = hashValue;
    }
  }
}

template <bool kMix, bool kHasNulls>
void hashDenseInt32(
    const int32_t* values,
    const uint64_t* nullBits,
    int32_t begin,
    int32_t end,
    uint64_t* output) {
  constexpr int kLanes = 8;
  int32_t row = begin;
  for (; row + kLanes <= end; row += kLanes) {
    uint32_t mixed[8];
    for (int lane = 0; lane < kLanes; ++lane) {
      mixed[lane] = static_cast<uint32_t>(values[row + lane]);
    }
    jenkinsRevMix32x8(mixed);
    uint64_t hashed[8];
    for (int lane = 0; lane < kLanes; ++lane) {
      if (kHasNulls && bits::isBitNull(nullBits, row + lane)) {
        hashed[lane] = BaseVector::kNullHash;
      } else {
        hashed[lane] = mixed[lane];
      }
    }
    storeHashes4<kMix>(output + row, hashed);
    storeHashes4<kMix>(output + row + 4, hashed + 4);
  }
  for (; row < end; ++row) {
    uint64_t hashValue;
    if constexpr (kHasNulls) {
      if (bits::isBitNull(nullBits, row)) {
        hashValue = BaseVector::kNullHash;
      } else {
        hashValue = folly::hasher<int32_t>()(values[row]);
      }
    } else {
      hashValue = folly::hasher<int32_t>()(values[row]);
    }
    if constexpr (kMix) {
      output[row] = bits::hashMix(output[row], hashValue);
    } else {
      output[row] = hashValue;
    }
  }
}

template <typename T, bool kMix, bool kHasNulls>
void hashDenseSmall(
    const T* values,
    const uint64_t* nullBits,
    int32_t begin,
    int32_t end,
    uint64_t* output) {
  for (int32_t row = begin; row < end; ++row) {
    uint64_t hashValue;
    if constexpr (kHasNulls) {
      if (bits::isBitNull(nullBits, row)) {
        hashValue = BaseVector::kNullHash;
      } else {
        hashValue = folly::hasher<T>()(values[row]);
      }
    } else {
      hashValue = folly::hasher<T>()(values[row]);
    }
    if constexpr (kMix) {
      output[row] = bits::hashMix(output[row], hashValue);
    } else {
      output[row] = hashValue;
    }
  }
}

template <typename T, bool kMix, bool kHasNulls>
void hashSelected(
    const T* values,
    const uint64_t* nullBits,
    const uint64_t* selectionBits,
    int32_t begin,
    int32_t end,
    uint64_t* output) {
  for (int32_t row = begin; row < end; ++row) {
    if (!bits::isBitSet(selectionBits, row)) {
      continue;
    }
    uint64_t hashValue;
    if constexpr (kHasNulls) {
      if (bits::isBitNull(nullBits, row)) {
        hashValue = BaseVector::kNullHash;
      } else {
        hashValue = folly::hasher<T>()(values[row]);
      }
    } else {
      hashValue = folly::hasher<T>()(values[row]);
    }
    if constexpr (kMix) {
      output[row] = bits::hashMix(output[row], hashValue);
    } else {
      output[row] = hashValue;
    }
  }
}

template <typename T, bool kMix, bool kHasNulls>
void hashDictionary(
    const T* baseValues,
    const int32_t* indices,
    const uint64_t* nullBits,
    int32_t begin,
    int32_t end,
    uint64_t* output) {
  constexpr int kLanes = 4;
  int32_t row = begin;
  if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
    for (; row + kLanes <= end; row += kLanes) {
      uint64_t hashed[4];
      for (int lane = 0; lane < kLanes; ++lane) {
        int32_t index = indices[row + lane];
        if (kHasNulls && bits::isBitNull(nullBits, row + lane)) {
          index = 0;
        }
        if constexpr (sizeof(T) == 8) {
          hashed[lane] = static_cast<uint64_t>(baseValues[index]);
        } else {
          hashed[lane] = static_cast<uint32_t>(baseValues[index]);
        }
      }
      if constexpr (sizeof(T) == 8) {
        twangMix64x4(hashed);
      } else {
        uint32_t narrow[8] = {};
        for (int lane = 0; lane < kLanes; ++lane) {
          narrow[lane] = static_cast<uint32_t>(hashed[lane]);
        }
        jenkinsRevMix32x8(narrow);
        for (int lane = 0; lane < kLanes; ++lane) {
          hashed[lane] = narrow[lane];
        }
      }
      if constexpr (kHasNulls) {
        applyNulls4(hashed, nullBits, row);
      }
      storeHashes4<kMix>(output + row, hashed);
    }
  }
  for (; row < end; ++row) {
    uint64_t hashValue;
    if constexpr (kHasNulls) {
      if (bits::isBitNull(nullBits, row)) {
        hashValue = BaseVector::kNullHash;
      } else {
        hashValue = folly::hasher<T>()(baseValues[indices[row]]);
      }
    } else {
      hashValue = folly::hasher<T>()(baseValues[indices[row]]);
    }
    if constexpr (kMix) {
      output[row] = bits::hashMix(output[row], hashValue);
    } else {
      output[row] = hashValue;
    }
  }
}

template <typename T, bool kMix>
bool tryHashTyped(
    DecodedVector& decoded,
    const SelectivityVector& rows,
    uint64_t* output) {
  const bool hasNulls = decoded.mayHaveNulls();
  const uint64_t* nullBits = hasNulls ? decoded.nulls(&rows) : nullptr;
  const auto begin = rows.begin();
  const auto end = rows.end();

  if (decoded.isIdentityMapping()) {
    const auto* values = decoded.data<T>();
    if (!values) {
      return false;
    }
    if (rows.isAllSelected()) {
      if constexpr (std::is_same_v<T, int64_t>) {
        hasNulls ? hashDenseInt64<kMix, true>(values, nullBits, begin, end, output)
                 : hashDenseInt64<kMix, false>(values, nullptr, begin, end, output);
      } else if constexpr (std::is_same_v<T, int32_t>) {
        hasNulls ? hashDenseInt32<kMix, true>(values, nullBits, begin, end, output)
                 : hashDenseInt32<kMix, false>(values, nullptr, begin, end, output);
      } else {
        hasNulls ? hashDenseSmall<T, kMix, true>(values, nullBits, begin, end, output)
                 : hashDenseSmall<T, kMix, false>(values, nullptr, begin, end, output);
      }
      return true;
    }
    if (rows.countSelected() < (end - begin) / kMinSelectivityDenominator) {
      return false;
    }
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t>) {
      const uint64_t* selectionBits = rows.allBits();
      hasNulls ? hashSelected<T, kMix, true>(
                     values, nullBits, selectionBits, begin, end, output)
               : hashSelected<T, kMix, false>(
                     values, nullptr, selectionBits, begin, end, output);
      return true;
    }
    return false;
  }

  if (!decoded.isConstantMapping() && rows.isAllSelected()) {
    const auto* indices = decoded.indices();
    if (!indices) {
      return false;
    }
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t>) {
      const auto* baseValues = decoded.data<T>();
      if (!baseValues) {
        return false;
      }
      hasNulls ? hashDictionary<T, kMix, true>(
                     baseValues, indices, nullBits, begin, end, output)
               : hashDictionary<T, kMix, false>(
                     baseValues, indices, nullptr, begin, end, output);
      return true;
    }
  }
  return false;
}

template <bool kMix, bool kHasNulls>
bool mapInt64RangeValueIds(
    const int64_t* values,
    const uint64_t* nullBits,
    int64_t rangeMin,
    uint64_t multiplier,
    int64_t rangeMax,
    int32_t begin,
    int32_t end,
    uint64_t* output) {
  constexpr int kLanes = 4;
  int32_t row = begin;
  const svbool_t pg = pgU64x4();
  for (; row + kLanes <= end; row += kLanes) {
    const svint64_t loaded = svld1_s64(pg, values + row);
    svbool_t outOfRange = svorr_b_z(
        pg,
        svcmplt_n_s64(pg, loaded, rangeMin),
        svcmpgt_n_s64(pg, loaded, rangeMax));
    uint8_t nullMask = 0;
    if constexpr (kHasNulls) {
      for (int lane = 0; lane < kLanes; ++lane) {
        if (bits::isBitNull(nullBits, row + lane)) {
          nullMask |= static_cast<uint8_t>(1u << lane);
        }
      }
      if (nullMask != 0) {
        svbool_t keep = pg;
        // Drop null lanes from the out-of-range test. Their ids are unused.
        uint64_t keepLane[4];
        for (int lane = 0; lane < kLanes; ++lane) {
          keepLane[lane] = (nullMask & (1u << lane)) ? 0 : 1;
        }
        keep = svcmpeq_n_u64(pg, svld1_u64(pg, keepLane), 1);
        outOfRange = svand_b_z(pg, outOfRange, keep);
      }
    }
    if (svptest_any(pg, outOfRange)) {
      return false;
    }

    const svuint64_t ids = svadd_n_u64_x(
        pg,
        svsub_u64_x(
            pg,
            svreinterpret_u64_s64(loaded),
            svdup_n_u64(static_cast<uint64_t>(rangeMin))),
        1);
    int64_t valueIds[4];
    svst1_s64(pg, valueIds, svreinterpret_s64_u64(ids));
    for (int lane = 0; lane < kLanes; ++lane) {
      if (kHasNulls && (nullMask & (1u << lane))) {
        if constexpr (!kMix) {
          output[row + lane] = 0;
        }
        continue;
      }
      const auto id = static_cast<uint64_t>(valueIds[lane]);
      if constexpr (kMix) {
        output[row + lane] += multiplier * id;
      } else if (multiplier == 1) {
        output[row + lane] = id;
      } else {
        output[row + lane] = output[row + lane] + multiplier * id;
      }
    }
  }

  for (; row < end; ++row) {
    if (kHasNulls && bits::isBitNull(nullBits, row)) {
      if (!kMix) {
        output[row] = 0;
      }
      continue;
    }
    const auto value = values[row];
    if (value < rangeMin || value > rangeMax) {
      return false;
    }
    const auto id =
        static_cast<uint64_t>(value) - static_cast<uint64_t>(rangeMin) + 1;
    if (kMix) {
      output[row] += multiplier * id;
    } else {
      output[row] = multiplier == 1 ? id : output[row] + multiplier * id;
    }
  }
  return true;
}

template <bool kMix, bool kHasNulls>
void rehashInt64(
    char** groups,
    int32_t columnOffset,
    int32_t numGroups,
    int32_t nullByte,
    uint8_t nullMask,
    uint64_t* hashes) {
  constexpr int kLanes = 4;
  if constexpr (!kMix) {
    for (int32_t i = 0; i < numGroups; ++i) {
      hashes[i] = folly::loadUnaligned<uint64_t>(groups[i] + columnOffset);
    }
    int32_t i = 0;
    for (; i + kLanes <= numGroups; i += kLanes) {
      twangMix64x4(hashes + i);
      if constexpr (kHasNulls) {
        for (int lane = 0; lane < kLanes; ++lane) {
          if (groups[i + lane][nullByte] & nullMask) {
            hashes[i + lane] = BaseVector::kNullHash;
          }
        }
      }
    }
    for (; i < numGroups; ++i) {
      if constexpr (kHasNulls) {
        if (groups[i][nullByte] & nullMask) {
          hashes[i] = BaseVector::kNullHash;
          continue;
        }
      }
      hashes[i] = folly::hasher<int64_t>()(static_cast<int64_t>(hashes[i]));
    }
    return;
  }

  constexpr int32_t kStackBuf = 1024;
  uint64_t stackBuf[kStackBuf];
  std::unique_ptr<uint64_t[]> heapBuf;
  uint64_t* loaded = stackBuf;
  if (numGroups > kStackBuf) {
    heapBuf = std::make_unique<uint64_t[]>(numGroups);
    loaded = heapBuf.get();
  }
  for (int32_t i = 0; i < numGroups; ++i) {
    loaded[i] = folly::loadUnaligned<uint64_t>(groups[i] + columnOffset);
  }
  int32_t i = 0;
  for (; i + kLanes <= numGroups; i += kLanes) {
    uint64_t hashed[4];
    for (int lane = 0; lane < kLanes; ++lane) {
      hashed[lane] = loaded[i + lane];
    }
    twangMix64x4(hashed);
    if constexpr (kHasNulls) {
      for (int lane = 0; lane < kLanes; ++lane) {
        if (groups[i + lane][nullByte] & nullMask) {
          hashed[lane] = BaseVector::kNullHash;
        }
      }
    }
    hashMix64x4(hashes + i, hashed);
  }
  for (; i < numGroups; ++i) {
    uint64_t hashValue;
    if constexpr (kHasNulls) {
      if (groups[i][nullByte] & nullMask) {
        hashValue = BaseVector::kNullHash;
      } else {
        hashValue = folly::hasher<int64_t>()(static_cast<int64_t>(loaded[i]));
      }
    } else {
      hashValue = folly::hasher<int64_t>()(static_cast<int64_t>(loaded[i]));
    }
    hashes[i] = bits::hashMix(hashes[i], hashValue);
  }
}

template <bool kMix, bool kHasNulls>
void rehashInt32(
    char** groups,
    int32_t columnOffset,
    int32_t numGroups,
    int32_t nullByte,
    uint8_t nullMask,
    uint64_t* hashes) {
  constexpr int kLanes = 8;
  constexpr int32_t kStackBuf = 1024;
  int32_t stackBuf[kStackBuf];
  std::unique_ptr<int32_t[]> heapBuf;
  int32_t* loaded = stackBuf;
  if (numGroups > kStackBuf) {
    heapBuf = std::make_unique<int32_t[]>(numGroups);
    loaded = heapBuf.get();
  }
  for (int32_t i = 0; i < numGroups; ++i) {
    loaded[i] = folly::loadUnaligned<int32_t>(groups[i] + columnOffset);
  }
  int32_t i = 0;
  for (; i + kLanes <= numGroups; i += kLanes) {
    uint32_t mixed[8];
    for (int lane = 0; lane < kLanes; ++lane) {
      mixed[lane] = static_cast<uint32_t>(loaded[i + lane]);
    }
    jenkinsRevMix32x8(mixed);
    uint64_t hashed[8];
    for (int lane = 0; lane < kLanes; ++lane) {
      if (kHasNulls && (groups[i + lane][nullByte] & nullMask)) {
        hashed[lane] = BaseVector::kNullHash;
      } else {
        hashed[lane] = mixed[lane];
      }
    }
    if constexpr (kMix) {
      hashMix64x4(hashes + i, hashed);
      hashMix64x4(hashes + i + 4, hashed + 4);
    } else {
      for (int lane = 0; lane < kLanes; ++lane) {
        hashes[i + lane] = hashed[lane];
      }
    }
  }
  for (; i < numGroups; ++i) {
    uint64_t hashValue;
    if constexpr (kHasNulls) {
      if (groups[i][nullByte] & nullMask) {
        hashValue = BaseVector::kNullHash;
      } else {
        hashValue = folly::hasher<int32_t>()(loaded[i]);
      }
    } else {
      hashValue = folly::hasher<int32_t>()(loaded[i]);
    }
    if constexpr (kMix) {
      hashes[i] = bits::hashMix(hashes[i], hashValue);
    } else {
      hashes[i] = hashValue;
    }
  }
}

template <typename T, bool kMix, bool kHasNulls>
void rehashSmallInteger(
    char** groups,
    int32_t columnOffset,
    int32_t numGroups,
    int32_t nullByte,
    uint8_t nullMask,
    uint64_t* hashes) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2);
  for (int32_t i = 0; i < numGroups; ++i) {
    uint64_t hashValue;
    if constexpr (kHasNulls) {
      if (groups[i][nullByte] & nullMask) {
        hashValue = BaseVector::kNullHash;
      } else {
        hashValue = folly::hasher<T>()(
            folly::loadUnaligned<T>(groups[i] + columnOffset));
      }
    } else {
      hashValue =
          folly::hasher<T>()(folly::loadUnaligned<T>(groups[i] + columnOffset));
    }
    if constexpr (kMix) {
      hashes[i] = bits::hashMix(hashes[i], hashValue);
    } else {
      hashes[i] = hashValue;
    }
  }
}

} // namespace

bool tryHash(
    VectorHasher& hasher,
    const SelectivityVector& rows,
    bool mixWithExisting,
    uint64_t* hashes) {
  auto& decoded = hasher.decodedVector();
  const auto kind = hasher.typeKind();
  if (kind == TypeKind::BIGINT) {
    return mixWithExisting ? tryHashTyped<int64_t, true>(decoded, rows, hashes)
                           : tryHashTyped<int64_t, false>(decoded, rows, hashes);
  }
  if (kind == TypeKind::INTEGER) {
    return mixWithExisting ? tryHashTyped<int32_t, true>(decoded, rows, hashes)
                           : tryHashTyped<int32_t, false>(decoded, rows, hashes);
  }
  if (kind == TypeKind::SMALLINT) {
    return mixWithExisting ? tryHashTyped<int16_t, true>(decoded, rows, hashes)
                           : tryHashTyped<int16_t, false>(decoded, rows, hashes);
  }
  if (kind == TypeKind::TINYINT) {
    return mixWithExisting ? tryHashTyped<int8_t, true>(decoded, rows, hashes)
                           : tryHashTyped<int8_t, false>(decoded, rows, hashes);
  }
  // REAL and DOUBLE keep NaNAwareHash in VectorHasher.
  return false;
}

bool tryValueIds(
    VectorHasher& hasher,
    const SelectivityVector& rows,
    bool mixWithExisting,
    uint64_t* hashes) {
  if (!hasher.isRange() || hasher.typeKind() != TypeKind::BIGINT) {
    return false;
  }
  auto& decoded = hasher.decodedVector();
  if (!decoded.isIdentityMapping() || !rows.isAllSelected()) {
    return false;
  }
  const auto* values = decoded.data<int64_t>();
  if (!values) {
    return false;
  }
  const bool hasNulls = decoded.mayHaveNulls();
  const uint64_t* nullBits = hasNulls ? decoded.nulls(&rows) : nullptr;
  auto call = [&](auto mixTag, auto nullTag) {
    return mapInt64RangeValueIds<
        decltype(mixTag)::value,
        decltype(nullTag)::value>(
        values,
        nullBits,
        hasher.rangeMin(),
        hasher.multiplier(),
        hasher.rangeMax(),
        rows.begin(),
        rows.end(),
        hashes);
  };
  if (mixWithExisting) {
    return hasNulls ? call(std::true_type{}, std::true_type{})
                    : call(std::true_type{}, std::false_type{});
  }
  return hasNulls ? call(std::false_type{}, std::true_type{})
                  : call(std::false_type{}, std::false_type{});
}

void rehashNormalizedKeys(
    char** groups,
    int32_t numGroups,
    uint64_t* hashes) {
  constexpr int kLanes = 4;
  for (int32_t i = 0; i < numGroups; ++i) {
    hashes[i] = RowContainer::normalizedKey(groups[i]);
  }
  int32_t i = 0;
  for (; i + kLanes <= numGroups; i += kLanes) {
    twangMix64x4(hashes + i);
  }
  for (; i < numGroups; ++i) {
    hashes[i] = folly::hasher<uint64_t>()(hashes[i]);
  }
}

bool tryRehash(
    TypeKind kind,
    char** groups,
    int32_t columnOffset,
    int32_t numGroups,
    bool mixWithExisting,
    int32_t nullByte,
    uint8_t nullMask,
    uint64_t* hashes) {
  const bool hasNulls = nullMask != 0;
  switch (kind) {
    case TypeKind::BIGINT:
      if (hasNulls) {
        mixWithExisting ? rehashInt64<true, true>(
                              groups, columnOffset, numGroups, nullByte, nullMask, hashes)
                        : rehashInt64<false, true>(
                              groups, columnOffset, numGroups, nullByte, nullMask, hashes);
      } else {
        mixWithExisting ? rehashInt64<true, false>(
                              groups, columnOffset, numGroups, 0, 0, hashes)
                        : rehashInt64<false, false>(
                              groups, columnOffset, numGroups, 0, 0, hashes);
      }
      return true;
    case TypeKind::INTEGER:
      if (hasNulls) {
        mixWithExisting ? rehashInt32<true, true>(
                              groups, columnOffset, numGroups, nullByte, nullMask, hashes)
                        : rehashInt32<false, true>(
                              groups, columnOffset, numGroups, nullByte, nullMask, hashes);
      } else {
        mixWithExisting ? rehashInt32<true, false>(
                              groups, columnOffset, numGroups, 0, 0, hashes)
                        : rehashInt32<false, false>(
                              groups, columnOffset, numGroups, 0, 0, hashes);
      }
      return true;
    case TypeKind::SMALLINT:
      if (hasNulls) {
        mixWithExisting
            ? rehashSmallInteger<int16_t, true, true>(
                  groups, columnOffset, numGroups, nullByte, nullMask, hashes)
            : rehashSmallInteger<int16_t, false, true>(
                  groups, columnOffset, numGroups, nullByte, nullMask, hashes);
      } else {
        mixWithExisting ? rehashSmallInteger<int16_t, true, false>(
                              groups, columnOffset, numGroups, 0, 0, hashes)
                        : rehashSmallInteger<int16_t, false, false>(
                              groups, columnOffset, numGroups, 0, 0, hashes);
      }
      return true;
    case TypeKind::TINYINT:
      if (hasNulls) {
        mixWithExisting
            ? rehashSmallInteger<int8_t, true, true>(
                  groups, columnOffset, numGroups, nullByte, nullMask, hashes)
            : rehashSmallInteger<int8_t, false, true>(
                  groups, columnOffset, numGroups, nullByte, nullMask, hashes);
      } else {
        mixWithExisting ? rehashSmallInteger<int8_t, true, false>(
                              groups, columnOffset, numGroups, 0, 0, hashes)
                        : rehashSmallInteger<int8_t, false, false>(
                              groups, columnOffset, numGroups, 0, 0, hashes);
      }
      return true;
    default:
      return false;
  }
}

} // namespace bytedance::bolt::exec::hash_table_simd
