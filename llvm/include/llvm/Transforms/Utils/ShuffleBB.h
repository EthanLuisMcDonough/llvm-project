//===-- ShuffleBB.h - Reorder Basic Blocks ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SHUFFLEBB_H
#define LLVM_TRANSFORMS_UTILS_SHUFFLEBB_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class ShuffleBasicBlocksPass : public PassInfoMixin<ShuffleBasicBlocksPass> {

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SHUFFLEBB_H
