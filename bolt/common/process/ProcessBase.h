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

#include <pthread.h>
#include <sys/types.h>

#include <cstdint>
#include <string>
#include <vector>
namespace bytedance {
namespace bolt {
namespace process {

/**
 * Current executable's name.
 */
std::string getAppName();

/**
 * This machine'a name.
 */
std::string getHostName();

/**
 * Process identifier.
 */
pid_t getProcessId();

/**
 * Current thread's identifier.
 */
pthread_t getThreadId();

/**
 * Get current working directory.
 */
std::string getCurrentDirectory();

/**
 * Returns elapsed CPU nanoseconds on the calling thread
 */
uint64_t threadCpuNanos();

// True if the machine has Intel AVX2 instructions and these are not disabled by
// flag.
bool hasAvx2();

// True on Linux aarch64 when SVE is enabled and the vector length is 32 bytes
// (256 bits). The check reads HWCAP before executing any SVE instruction.
bool hasSve256();

// True when the optional SIMD hash table in exec/HashTable can run: AVX2, or
// 256-bit SVE. Other SVE widths stay off; those paths are fixed at 4x u64.
inline bool hasSimdHashTableIsa() {
  return hasAvx2() || hasSve256();
}

// True if the machine has Intel BMI2 instructions and these are not disabled by
// flag.
bool hasBmi2();

// True if the machine has ARM Neon instructions
bool hasNeon();

} // namespace process
} // namespace bolt
} // namespace bytedance
