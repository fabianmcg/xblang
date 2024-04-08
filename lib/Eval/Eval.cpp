//===- Codegen.cpp - Code generation -----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines code generation functions and classes.
//
//===----------------------------------------------------------------------===//

#include "xblang/Eval/Eval.h"
#include "xblang/Basic/ContextDialect.h"
#include "xblang/Basic/PatternApplicator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace xblang;
using namespace xblang::eval;
using namespace xblang::eval::detail;

#define DEBUG_TYPE "eval"

namespace xblang {
namespace eval {
namespace detail {
class EvalDriverImpl : public mlir::RewriterBase::Listener {
public:
  EvalDriverImpl(EvalDriver *driver, const FrozenPatternSet &patterns);
  /// Returns the matched pattern.
  const EvalPattern *getPattern(Operation *op);
  /// Invokes the eval method for Op.
  Attribute eval(Operation *op, ArrayRef<Attribute> args);

private:
  // Pattern applicator.
  GenericPatternApplicator applicator;
  // Logger.
  llvm::ScopedPrinter logger{llvm::dbgs()};
  // Pointer to the main driver.
  EvalDriver *driver;
  // Logs a result to the logger.
  void logResult(StringRef result, const llvm::Twine &msg = {},
                 bool isSuccess = false, Attribute value = nullptr);
};
} // namespace detail
} // namespace eval
} // namespace xblang

//===----------------------------------------------------------------------===//
// EvalDriverImpl
//===----------------------------------------------------------------------===//
EvalDriverImpl::EvalDriverImpl(EvalDriver *driver,
                               const FrozenPatternSet &patterns)
    : applicator(patterns), driver(driver) {
  applicator.applyDefaultCostModel();
}

void EvalDriverImpl::logResult(StringRef result, const llvm::Twine &msg,
                               bool isSuccess, Attribute value) {
  logger.unindent();
  logger.startLine() << "} -> " << result;
  if (!msg.isTriviallyEmpty())
    logger.getOStream() << " : " << msg;
  if (isSuccess) {
    logger.getOStream() << ", value = ";
    if (value)
      value.print(logger.getOStream());
    else
      logger.getOStream() << "<<NULL>>";
  }
  logger.getOStream() << "\n";
}

const EvalPattern *EvalDriverImpl::getPattern(Operation *op) {
#ifndef NDEBUG
  auto onFailure = [&](const Pattern &pattern) {
    LLVM_DEBUG({
      logger.getOStream() << "\n";
      logger.startLine() << "* Eval pattern " << pattern.getDebugName()
                         << " : '" << op->getName() << " -> (";
      llvm::interleaveComma(pattern.getGeneratedOps(), logger.getOStream());
      logger.getOStream() << ")' {\n";
      logger.indent();
      logResult("failure", "pattern failed to match");
    });
  };
#else
  function_ref<void(const Pattern &)> onFailure = {};
#endif
  return applicator.getPattern<EvalPattern>(op, {}, onFailure);
}

Attribute EvalDriverImpl::eval(Operation *op, ArrayRef<Attribute> args) {
  Attribute result = nullptr;
#ifndef NDEBUG
  auto canApply = [&](const Pattern &pattern) {
    LLVM_DEBUG({
      logger.getOStream() << "\n";
      logger.startLine() << "* Pattern " << pattern.getDebugName() << " : '"
                         << op->getName() << " -> (";
      llvm::interleaveComma(pattern.getGeneratedOps(), logger.getOStream());
      logger.getOStream() << ")' {\n";
      logger.indent();
    });
    return true;
  };
  auto onSuccess = [&](const Pattern &pattern) {
    LLVM_DEBUG(
        logResult("success", "pattern matched successfully", true, result));
  };
#endif
  if (const EvalPattern *pattern = getPattern(op)) {
#ifndef NDEBUG
    canApply(*pattern);
#endif
    result = pattern->eval(op, args, *driver);
#ifndef NDEBUG
    onSuccess(*pattern);
#endif
  }
  return result;
}

//===----------------------------------------------------------------------===//
// CodegenRewriter
//===----------------------------------------------------------------------===//

EvalDriver::EvalDriver(XBContext *context, const FrozenPatternSet &patterns)
    : Builder(context->getMLIRContext()), context(context) {
  impl.reset(new EvalDriverImpl(this, patterns));
}

EvalDriver::~EvalDriver() = default;

Attribute EvalDriver::eval(Operation *op, ArrayRef<Attribute> args) {
  return impl->eval(op, args);
}

//===----------------------------------------------------------------------===//
// Driver functions
//===----------------------------------------------------------------------===//

Attribute xblang::eval::evaluate(Operation *op, ArrayRef<Attribute> args,
                                 const FrozenPatternSet &patterns) {
  XBContextDialect *dialect =
      op->getContext()->getLoadedDialect<XBContextDialect>();
  if (!dialect)
    return nullptr;
  return EvalDriver(&dialect->getContext(), patterns).eval(op, args);
}
