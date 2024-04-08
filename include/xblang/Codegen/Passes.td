//===- Passes.td - Code generation passes ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_PASSES
#define CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

def Codegen : Pass<"xblang-codegen"> {
  let summary = "Performs the code generation pass.";
  let description = [{
  }];
}

#endif // CODEGEN_PASSES
