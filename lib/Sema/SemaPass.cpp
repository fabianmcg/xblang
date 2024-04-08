//===- SemaPass.cpp - Semantic checker pass ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the semantic checker pass.
//
//===----------------------------------------------------------------------===//

#include "xblang/Basic/ContextDialect.h"
#include "xblang/Eval/Eval.h"
#include "xblang/Interfaces/Eval.h"
#include "xblang/Interfaces/Sema.h"
#include "xblang/Sema/Passes.h"
#include "xblang/Sema/Sema.h"
#include "xblang/XLG/IR/XLGDialect.h"

namespace xblang {
namespace sema {
#define GEN_PASS_DEF_SEMACHECKER
#include "xblang/Sema/Passes.h.inc"
} // namespace sema
} // namespace xblang

using namespace xblang;
using namespace xblang::sema;

namespace {
struct SemaChecker : public xblang::sema::impl::SemaCheckerBase<SemaChecker> {
  using Base::Base;

  void runOnOperation() override;
};

struct BuiltinModuleVerifier : public SemaOpPattern<mlir::ModuleOp> {
  using Base::Base;

  SemaResult check(Op op, Status status, SymbolTable *symTable,
                   SemaDriver &driver) const final {
    return driver.checkRegions(op, symTable);
  }
};
} // namespace

void SemaChecker::runOnOperation() {
  auto *dialect = getContext().getLoadedDialect<XBContextDialect>();
  if (!dialect)
    return;
  GenericPatternSet set(&(dialect->getContext()));
  set.add<BuiltinModuleVerifier>(set.getMLIRContext());
  mlir::DialectInterfaceCollection<SemaDialectInterface> collection(
      &getContext());
  TypeSystem typeSystem;
  for (auto &interface : collection)
    interface.populateSemaPatterns(set, typeSystem);
  mlir::DialectInterfaceCollection<EvalDialectInterface> evalCollection(
      &getContext());
  GenericPatternSet evalPatterns(&(dialect->getContext()));
  for (auto &interface : evalCollection)
    interface.populateEvalPatterns(evalPatterns);
  eval::EvalDriver evalDriver(&(dialect->getContext()),
                              FrozenPatternSet(evalPatterns));
  if (mlir::failed(applySemaDriver(getOperation(), FrozenPatternSet(set),
                                   typeSystem, &evalDriver)))
    return signalPassFailure();
}
