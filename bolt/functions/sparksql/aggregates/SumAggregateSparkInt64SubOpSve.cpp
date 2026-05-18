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
 *
 * AArch64 SVE batch kernel for Spark sum(bigint) HashAgg group updates.
 * Compiled only on aarch64 (see aggregates/CMakeLists.txt, `-march=armv8-a+sve`).
 */

#include "bolt/functions/sparksql/aggregates/SumAggregateSparkInt64SubOp.h"

#include <arm_sve.h>

#include "bolt/vector/BaseVector.h"
#include "bolt/vector/DecodedVector.h"
#include "bolt/vector/SelectivityVector.h"

namespace bytedance::bolt::functions::aggregate::sparksql {

namespace {

constexpr uint64_t kSupportedSveVectorBytes = 32;

template <typename T>
inline bool isBitSet(const T* bits, uint64_t idx) {
  return bits[idx / (sizeof(bits[0]) * 8)] &
      (static_cast<T>(1) << (idx & ((sizeof(bits[0]) * 8) - 1)));
}

inline bool isBitNull(const uint64_t* bits, int32_t index) {
  return isBitSet(bits, index) == false;
}
template <typename T, typename U>
constexpr inline T roundUp(T value, U factor) {
  return (value + (factor - 1)) / factor * factor;
}

svbool_t sveDecodedNullMaskForMode(
    uint8_t* nulls_,
    int32_t index,
    int mode,
    uint32_t* dic,
    int32_t length) {
  svbool_t pg;
  if (mode == 0) {
    pg = svptrue_b8();
    return pg;
  } else if (mode == 1) {
    __asm__ __volatile__("ldr %0, [%1]"
                         : "=Upl"(pg)
                         : "r"(&(nulls_[index]))
                         : "memory");
    return pg;
  } else if (mode == 2) {
    if (!isBitNull(
            reinterpret_cast<uint64_t*>(nulls_),
            0))
    {
      pg = svptrue_b8();
    } else {
      pg = svpfalse();
    }
    return pg;
  } else if (mode == 3) {

    svuint32_t onc = svdup_u32(1);
    svuint32_t inv = svindex_u32(0, 1);
    svuint32_t pow = svlsl_m(svptrue_b32(), onc, inv);
    uint8_t tmpNulls[4] = {0};
    uint32_t* null32ptr = reinterpret_cast<uint32_t*>(nulls_);

    svuint32_t posv, idxbufv, bufv, offsetv;
    svbool_t nullvec, pg1;

    // mode1==3: pack null bits for eight dictionary lanes (chunk 0).
    pg1 = svwhilelt_b32(index * 8, length);
    posv = svld1(pg1, dic + index * 8);
    idxbufv = svlsr_x(pg1, posv, 5); // u32 word index (pos / 32)
    bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
    offsetv = svand_m(pg1, posv, 0b11111); // bit index within the u32 word
    bufv = svlsr_m(pg1, bufv, offsetv);
    bufv = svand_m(pg1, bufv, 0x1);
    nullvec = svcmpgt(pg1, bufv, 0);
    if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
      uint8_t nullsres = svaddv(nullvec, pow);
      tmpNulls[0] = nullsres;
    } else {
      tmpNulls[0] = 0;
    }

    // mode1==3: dictionary null bits (chunk 1).
    pg1 = svwhilelt_b32(index * 8 + 8, length);
    posv = svld1(pg1, dic + index * 8 + 8);
    idxbufv = svlsr_x(pg1, posv, 5);
    bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
    offsetv = svand_m(pg1, posv, 0b11111);
    bufv = svlsr_m(pg1, bufv, offsetv);
    bufv = svand_m(pg1, bufv, 0x1);
    nullvec = svcmpgt(pg1, bufv, 0);
    if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
      uint8_t nullsres = svaddv(nullvec, pow);
      tmpNulls[1] = nullsres;
    } else {
      tmpNulls[1] = 0;
    }

    // mode1==3: dictionary null bits (chunk 2).
    pg1 = svwhilelt_b32(index * 8 + 16, length);
    posv = svld1(pg1, dic + index * 8 + 16);
    idxbufv = svlsr_x(pg1, posv, 5);
    bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
    offsetv = svand_m(pg1, posv, 0b11111);
    bufv = svlsr_m(pg1, bufv, offsetv);
    bufv = svand_m(pg1, bufv, 0x1);
    nullvec = svcmpgt(pg1, bufv, 0);
    if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
      uint8_t nullsres = svaddv(nullvec, pow);
      tmpNulls[2] = nullsres;
    } else {
      tmpNulls[2] = 0;
    }

    // mode1==3: dictionary null bits (chunk 3).
    pg1 = svwhilelt_b32(index * 8 + 24, length);
    posv = svld1(pg1, dic + index * 8 + 24);
    idxbufv = svlsr_x(pg1, posv, 5);
    bufv = svld1_gather_index(pg1, null32ptr, idxbufv);
    offsetv = svand_m(pg1, posv, 0b11111);
    bufv = svlsr_m(pg1, bufv, offsetv);
    bufv = svand_m(pg1, bufv, 0x1);
    nullvec = svcmpgt(pg1, bufv, 0);
    if (__builtin_expect((svptest_any(pg1, nullvec)), 0)) {
      uint8_t nullsres = svaddv(nullvec, pow);
      tmpNulls[3] = nullsres;
    } else {
      tmpNulls[3] = 0;
    }

    __asm__ __volatile__("ldr %0, [%1]"
                         : "=Upl"(pg)
                         : "r"(tmpNulls)
                         : "memory");
    return pg;
  }
  // Unknown mode1: inactive predicate.
  pg = svpfalse();
  return pg;
}

inline __attribute__((always_inline)) svbool_t
sveMaskDistinctGroupSlots(svbool_t pg, const svuint64_t val) {
  svuint64_t s1 = svext_u64(val, val, 1);
  svbool_t mask2 = svcmpeq(svwhilelt_b64(0, 3), val, s1);

  svuint64_t s2 = svext_u64(val, val, 2);
  svbool_t mask3 = svcmpeq(svwhilelt_b64(0, 2), val, s2);
  svbool_t mask12 = svorr_b_z(pg, mask2, mask3);

  svuint64_t s3 = svext_u64(val, val, 3);
  svbool_t mask4 = svcmpeq(svwhilelt_b64(0, 1), val, s3);

  svbool_t mask = svorr_b_z(pg, mask4, mask12);
  mask = svnot_b_z(pg, mask);

  return mask;
}

static bool sveClearGroupNullFlags(
    int32_t nullByte,
    uint8_t nullMask,
    uint64_t* numNulls,
    svuint64_t ptr,
    svbool_t pg) {
  if (*numNulls) {
    svint64_t group =
        svld1sb_gather_u64base_offset_s64(pg, ptr, nullByte);
    svuint8_t group8 = svreinterpret_u8(group);

    svuint8_t tmp = svand_n_u8_z(pg, group8, nullMask);
    svbool_t test = svcmpne_n_u8(svptrue_b8(), tmp, 0);
    if (svptest_any(svptrue_b8(), test)) {
      uint8_t negNull = ~nullMask;

      svuint8_t adjust = svand_n_u8_m(test, group8, negNull);
      svst1b_scatter_u64base_offset_s64(
          pg, ptr, nullByte, svreinterpret_s64(adjust));

      int num = svcntp_b8(test, test);
      *numNulls -= num;
      return true;
    }
  }
  return false;
}

template <typename GetPtr>
static void sveHashAggBatchUpdateGroupSums(
      int32_t nullByte,
      uint8_t nullMask,
      uint64_t* numNulls,
      GetPtr&& getAccumPtr,
      char** result,
      uint64_t* bitmap1,
      uint64_t* bitmap2,
      int64_t* value,
      int32_t begin,
      int32_t end,
      int mode1,
      int mode2,
      uint32_t* dic) {
  uint8_t* bitmap1_8 = reinterpret_cast<uint8_t*>(bitmap1);
  uint8_t* bitmap2_8 = reinterpret_cast<uint8_t*>(bitmap2);

  int32_t firstWord =
      roundUp(begin, 32) == begin ? begin : roundUp(begin, 32) - 32;
  int32_t lastWord = roundUp(end, 32);
  svbool_t mask, mask1;
  // Process 32 logical rows per iteration; `count` is the row index.
  for (int32_t count = firstWord; count + 32 <= lastWord; count += 32) {
    int32_t arr8Index = count / 8;
    svbool_t mask2;
    if (bitmap2_8 != nullptr) {
      mask2 = sveDecodedNullMaskForMode(bitmap2_8, arr8Index, mode1, dic, end);
    } else {
      mask2 = svptrue_b8();
    }
    __asm__ __volatile__("ldr %0, [%1]"
                         : "=Upl"(mask1)
                         : "r"(&bitmap1_8[arr8Index])
                         : "memory");
    mask = svand_b_z(svptrue_b8(), mask1, mask2);
    mask = svand_b_z(svptrue_b8(), mask, svwhilelt_b8(count, end));
    if (!svptest_any(svptrue_b8(), mask)) {
      continue;
    }

    svbool_t mask00 = svunpklo(mask);
    svbool_t mask01 = svunpkhi(mask);
    if (svptest_any(svptrue_b16(), mask00)) {
      svbool_t mask10 = svunpklo(mask00);
      if (svptest_any(svptrue_b32(), mask10)) {
        svbool_t mask20 = svunpklo(mask10);
        svbool_t mask21 = svunpkhi(mask10);
        if (svptest_any(svptrue_b64(), mask20)) {
          svuint64_t ptr =
              svld1(mask20, reinterpret_cast<uint64_t*>(result + count));
          svbool_t m20 = sveMaskDistinctGroupSlots(mask20, ptr);
          sveClearGroupNullFlags(nullByte, nullMask, numNulls, ptr, m20);
          uint8_t flag0[4] = {0, 0, 0, 0};
          __asm__ __volatile__("str %1, [%0]": : "r" (&flag0[0]), "Upl" (mask20) : "memory");
          
          // mode2==3: gather via dictionary indices; else flat row values.
          if (mode2 == 3) {
            for (int i = 0; i < 4; i++) {
              if (flag0[i] != 0) {
                uint32_t dictIndex = dic[count + i];
                int64_t dictValue = value[dictIndex];
                *getAccumPtr(*(result + count + i)) += dictValue;
              }
            }
          } else {
            for (int i = 0; i < 4; i++) {
              if (flag0[i] != 0) {
                *getAccumPtr(*(result + count + i)) += value[count + i];
              }
            }
          }
        }

        if (svptest_any(svptrue_b64(), mask21)) {
          svuint64_t ptr =
              svld1(mask21, reinterpret_cast<uint64_t*>(result + count + 4));
          svbool_t m21 = sveMaskDistinctGroupSlots(mask21, ptr);
          sveClearGroupNullFlags(nullByte, nullMask, numNulls, ptr, m21);
          uint8_t flag1[4] = {0, 0, 0, 0};
          __asm__ __volatile__("str %1, [%0]": : "r" (&flag1[0]), "Upl" (mask21) : "memory");
          
          if (mode2 == 3) {
            for (int i = 0; i < 4; i++) {
              if (flag1[i] != 0) {
                uint32_t dictIndex = dic[count + 4 + i];
                int64_t dictValue = value[dictIndex];
                *getAccumPtr(*(result + count + 4 + i)) += dictValue;
              }
            }
          } else {
            for (int i = 0; i < 4; i++) {
              if (flag1[i] != 0) {
                *getAccumPtr(*(result + count + 4 + i)) += value[count + 4 + i];
              }
            }
          }
        }
      }
      svbool_t mask11 = svunpkhi(mask00);
      if (svptest_any(svptrue_b32(), mask11)) {
        svbool_t mask22 = svunpklo(mask11);
        svbool_t mask23 = svunpkhi(mask11);
        if (svptest_any(svptrue_b64(), mask22)) {
          svuint64_t ptr =
              svld1(mask22, reinterpret_cast<uint64_t*>(result + count + 8));
          svbool_t m22 = sveMaskDistinctGroupSlots(mask22, ptr);
          sveClearGroupNullFlags(nullByte, nullMask, numNulls, ptr, m22);
          uint8_t flag2[4] = {0, 0, 0, 0};
          __asm__ __volatile__("str %1, [%0]": : "r" (&flag2[0]), "Upl" (mask22) : "memory");
          
          if (mode2 == 3) {
            for (int i = 0; i < 4; i++) {
              if (flag2[i] != 0) {
                uint32_t dictIndex = dic[count + 8 + i];
                int64_t dictValue = value[dictIndex];
                *getAccumPtr(*(result + count + 8 + i)) += dictValue;
              }
            }
          } else {
            for (int i = 0; i < 4; i++) {
              if (flag2[i] != 0) {
                *getAccumPtr(*(result + count + 8 + i)) += value[count + 8 + i];
              }
            }
          }
        }

        if (svptest_any(svptrue_b64(), mask23)) {
          svuint64_t ptr =
              svld1(mask23, reinterpret_cast<uint64_t*>(result + count + 12));
          svbool_t m23 = sveMaskDistinctGroupSlots(mask23, ptr);
          sveClearGroupNullFlags(nullByte, nullMask, numNulls, ptr, m23);
          uint8_t flag3[4] = {0, 0, 0, 0};
          __asm__ __volatile__("str %1, [%0]": : "r" (&flag3[0]), "Upl" (mask23) : "memory");
          
          if (mode2 == 3) {
            for (int i = 0; i < 4; i++) {
              if (flag3[i] != 0) {
                uint32_t dictIndex = dic[count + 12 + i];
                int64_t dictValue = value[dictIndex];
                *getAccumPtr(*(result + count + 12 + i)) += dictValue;
              }
            }
          } else {
            for (int i = 0; i < 4; i++) {
              if (flag3[i] != 0) {
                *getAccumPtr(*(result + count + 12 + i)) += value[count + 12 + i];
              }
            }
          }
        }
      }
    }

    svbool_t mask12 = svunpklo(mask01);

    if (svptest_any(svptrue_b16(), mask01)) {
      svbool_t mask24 = svunpklo(mask12);
      svbool_t mask25 = svunpkhi(mask12);
      if (svptest_any(svptrue_b32(), mask12)) {
        if (svptest_any(svptrue_b64(), mask24)) {
          svuint64_t ptr =
              svld1(mask24, reinterpret_cast<uint64_t*>(result + count + 16));
          svbool_t m24 = sveMaskDistinctGroupSlots(mask24, ptr);
          sveClearGroupNullFlags(nullByte, nullMask, numNulls, ptr, m24);
          uint8_t flag4[4] = {0, 0, 0, 0};
          __asm__ __volatile__("str %1, [%0]": : "r" (&flag4[0]), "Upl" (mask24) : "memory");
          
          if (mode2 == 3) {
            for (int i = 0; i < 4; i++) {
              if (flag4[i] != 0) {
                uint32_t dictIndex = dic[count + 16 + i];
                int64_t dictValue = value[dictIndex];
                *getAccumPtr(*(result + count + 16 + i)) += dictValue;
              }
            }
          } else {
            for (int i = 0; i < 4; i++) {
              if (flag4[i] != 0) {
                *getAccumPtr(*(result + count + 16 + i)) += value[count + 16 + i];
              }
            }
          }
        }

        if (svptest_any(svptrue_b64(), mask25)) {
          svuint64_t ptr =
              svld1(mask25, reinterpret_cast<uint64_t*>(result + count + 20));
          svbool_t m25 = sveMaskDistinctGroupSlots(mask25, ptr);
          sveClearGroupNullFlags(nullByte, nullMask, numNulls, ptr, m25);
          uint8_t flag5[4] = {0, 0, 0, 0};
          __asm__ __volatile__("str %1, [%0]": : "r" (&flag5[0]), "Upl" (mask25) : "memory");
          
          if (mode2 == 3) {
            for (int i = 0; i < 4; i++) {
              if (flag5[i] != 0) {
                uint32_t dictIndex = dic[count + 20 + i];
                int64_t dictValue = value[dictIndex];
                *getAccumPtr(*(result + count + 20 + i)) += dictValue;
              }
            }
          } else {
            for (int i = 0; i < 4; i++) {
              if (flag5[i] != 0) {
                *getAccumPtr(*(result + count + 20 + i)) += value[count + 20 + i];
              }
            }
          }
        }
      }
      svbool_t mask13 = svunpkhi(mask01);

      if (svptest_any(svptrue_b32(), mask13)) {
        svbool_t mask26 = svunpklo(mask13);
        svbool_t mask27 = svunpkhi(mask13);
        if (svptest_any(svptrue_b64(), mask26)) {
          svuint64_t ptr =
              svld1(mask26, reinterpret_cast<uint64_t*>(result + count + 24));
          svbool_t m26 = sveMaskDistinctGroupSlots(mask26, ptr);
          sveClearGroupNullFlags(nullByte, nullMask, numNulls, ptr, m26);
          uint8_t flag6[4] = {0, 0, 0, 0};
          __asm__ __volatile__("str %1, [%0]": : "r" (&flag6[0]), "Upl" (mask26) : "memory");
          
          if (mode2 == 3) {
            for (int i = 0; i < 4; i++) {
              if (flag6[i] != 0) {
                uint32_t dictIndex = dic[count + 24 + i];
                int64_t dictValue = value[dictIndex];
                *getAccumPtr(*(result + count + 24 + i)) += dictValue;
              }
            }
          } else {
            for (int i = 0; i < 4; i++) {
              if (flag6[i] != 0) {
                *getAccumPtr(*(result + count + 24 + i)) += value[count + 24 + i];
              }
            }
          }
        }

        if (svptest_any(svptrue_b64(), mask27)) {
          svuint64_t ptr =
              svld1(mask27, reinterpret_cast<uint64_t*>(result + count + 28));
          svbool_t m27 = sveMaskDistinctGroupSlots(mask27, ptr);
          sveClearGroupNullFlags(nullByte, nullMask, numNulls, ptr, m27);
          uint8_t flag7[4] = {0, 0, 0, 0};
          __asm__ __volatile__("str %1, [%0]": : "r" (&flag7[0]), "Upl" (mask27) : "memory");
          
          if (mode2 == 3) {
            for (int i = 0; i < 4; i++) {
              if (flag7[i] != 0) {
                uint32_t dictIndex = dic[count + 28 + i];
                int64_t dictValue = value[dictIndex];
                *getAccumPtr(*(result + count + 28 + i)) += dictValue;
              }
            }
          } else {
            for (int i = 0; i < 4; i++) {
              if (flag7[i] != 0) {
                *getAccumPtr(*(result + count + 28 + i)) += value[count + 28 + i];
              }
            }
          }
        }
      }
    }
  }
}

} // namespace

bool SumAggregateSparkInt64SubOp::updateGroupsFromDecoded(
    char** groups,
    const SelectivityVector& rows,
    ::bytedance::bolt::DecodedVector& decoded) {
  // This kernel stores SVE predicates into four-byte scratch buffers and
  // processes 32 rows per block. Use it only on 256-bit SVE; other VLs fall
  // back to the scalar Base path in the caller.
  if (svcntb() != kSupportedSveVectorBytes) {
    return false;
  }

  const int32_t mode1 = decoded.hashAggNullsLayoutMode();
  const int32_t mode2 = decoded.hashAggIndicesLayoutMode();
  if (mode2 == 2) {
    return false;
  }
  uint64_t* bitmap2 = decoded.hashAggMutableCombinedNullBits();
  int64_t* valueBuf = reinterpret_cast<int64_t*>(decoded.hashAggMutableRawData());
  uint32_t* dic = reinterpret_cast<uint32_t*>(decoded.hashAggMutableIndices());

  uint64_t* rowsBits = const_cast<uint64_t*>(rows.allBits());
  const vector_size_t begin = rows.begin();
  const vector_size_t end = rows.end();

  auto getAccum = [this](char* group) -> int64_t* {
    return this->template value<int64_t>(group);
  };

  // Constant index layout (mode2==2): non-null constants stay on Base because
  // mayHaveNulls() is false. Nullable null constants (mode1==2) that reach SVE
  // rely on sveDecodedNullMaskForMode to clear the block mask (no value adds).

  sveHashAggBatchUpdateGroupSums(
      nullByte_,
      nullMask_,
      &numNulls_,
      getAccum,
      groups,
      rowsBits,
      bitmap2,
      valueBuf,
      begin,
      end,
      mode1,
      mode2,
      dic);
  // On aarch64: batch handled by SVE; caller skips Base. Shape gating is upstream
  // (`mayHaveNulls`, `numNulls_`, auxv).
  return true;
}

} // namespace bytedance::bolt::functions::aggregate::sparksql
