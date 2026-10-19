/*
 * Copyright (c) ByteDance Ltd. and/or its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <arm_sve.h>

#include <cstdint>

namespace bytedance::bolt::process {

uint64_t sveVectorLengthBytes() {
  return svcntb();
}

} // namespace bytedance::bolt::process
