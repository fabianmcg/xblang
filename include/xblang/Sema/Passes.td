//===- Passes.td - Semantic checker passes -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SEMA_PASSES
#define SEMA_PASSES

include "mlir/Pass/PassBase.td"

def SemaChecker : Pass<"xblang-sema"> {
  let summary = "Performs semantic checking on the IR.";
  let description = [{
  }];
}

#endif // SEMA_PASSES
