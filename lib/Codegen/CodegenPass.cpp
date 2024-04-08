//===- CodegenPass.cpp - Code generation pass --------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the code generation pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "xblang/Basic/ContextDialect.h"
#include "xblang/Basic/Pattern.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Codegen/Passes.h"
#include "xblang/Interfaces/Codegen.h"

namespace xblang {
namespace codegen {
#define GEN_PASS_DEF_CODEGEN
#include "xblang/Codegen/Passes.h.inc"
} // namespace codegen
} // namespace xblang

using namespace xblang;
using namespace xblang::codegen;

namespace {
struct Codegen : public xblang::codegen::impl::CodegenBase<Codegen> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void Codegen::runOnOperation() {
  auto *dialect = getContext().getLoadedDialect<XBContextDialect>();
  if (!dialect)
    return;
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  GenericPatternSet patterns(&dialect->getContext());
  mlir::DialectInterfaceCollection<CodegenDialectInterface> collection(
      &getContext());
  for (auto &interface : collection)
    if (failed(interface.populateCodegenPatterns(patterns, &converter)))
      return signalPassFailure();
  converter.addConversion([&](FunctionType type) {
    SmallVector<Type, 8> ins(type.getInputs());
    SmallVector<Type, 8> outs(type.getResults());
    for (auto &in : ins)
      in = converter.convertType(in);
    for (auto &out : outs)
      out = converter.convertType(out);
    return FunctionType::get(&getContext(), ins, outs);
  });
  FrozenPatternSet patternSet(patterns);
  if (mlir::failed(applyCodegenDriver(getOperation(), patternSet, &converter)))
    return signalPassFailure();
}
