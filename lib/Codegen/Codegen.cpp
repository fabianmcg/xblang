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

#include "xblang/Codegen/Codegen.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xblang/Basic/ContextDialect.h"
#include "xblang/Basic/PatternApplicator.h"
#include "xblang/Codegen/Utils.h"
#include "xblang/Sema/TypeUtil.h"
#include "xblang/Support/Worklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace xblang;
using namespace xblang::codegen;
using namespace xblang::codegen::detail;

#define DEBUG_TYPE "codegen"

namespace {
bool isPrimitiveCast(Type target, Type source) {
  return isPrimitiveType(target) && isPrimitiveType(source);
}

struct IRManager {
  /// Processes the worklists.
  LogicalResult cleanup(Location loc);
  /// Erases an operation.
  void deleteOp(Operation *op);
  /// Erases an operation.
  void eraseOp(Operation *op);
  /// Erases a block.
  void eraseBlock(Block *block);
  // Worklist of operations to be erased.
  SmallVector<Operation *> opWorklist;
  // Worklist of blocks to be erased.
  SmallVector<Block *> blockWorklist;
  // Set of erased elements.
  DenseSet<void *> erased;
};
} // namespace

namespace xblang {
namespace codegen {
namespace detail {
class CGDriverImpl : public mlir::RewriterBase::Listener {
public:
  template <typename T>
  using Queue = std::deque<T>;
  using Worklist = GenericIRWorklist<Queue, IRWorklistElement>;
  CGDriverImpl(CGDriver *driver, const FrozenPatternSet &patterns);
  // Adds an operation to the worklist.
  void addToWorklist(Operation *op);
  // Process the elements of the worklist.
  LogicalResult processWorklist();
  // Runs the driver.
  LogicalResult run(Operation *op);

  /// Notify the driver that the specified operation may have been modified
  /// in-place. The operation is added to the worklist.
  void notifyOperationModified(Operation *op) override;

  /// Notify the driver that the specified operation was inserted. Update the
  /// worklist as needed: The operation is enqueued depending on scope and
  /// strict mode.
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;

  /// Notify the driver that the specified operation was removed. Update the
  /// worklist as needed: The operation and its children are removed from the
  /// worklist.
  void notifyOperationErased(Operation *op) override;

  /// Notify the driver that the specified operation was replaced. Update the
  /// worklist as needed: New users are added enqueued.
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;

  /// Notify the driver that the given block was inserted.
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override;

  /// Notify the driver that the given block is about to be removed.
  void notifyBlockErased(Block *block) override;

  /// Returns the matched pattern.
  const CGPattern *getPattern(Operation *op);

  /// Invokes the generate method for Op.
  CGResult genOp(Operation *op);

private:
  Operation *rootOp{};
  // IR manager.
  IRManager irManager;
  // Pattern applicator.
  GenericPatternApplicator applicator;
  // Op worklist.
  Worklist worklist{};
  // Logger.
  llvm::ScopedPrinter logger{llvm::dbgs()};
  // Pointer to the main driver.
  CGDriver *driver;
  // Logs a result to the logger.
  void logResult(StringRef result, const llvm::Twine &msg = {});
};
} // namespace detail
} // namespace codegen
} // namespace xblang

//===----------------------------------------------------------------------===//
// IRManager
//===----------------------------------------------------------------------===//

LogicalResult IRManager::cleanup(Location loc) {
  for (Operation *op : llvm::reverse(opWorklist)) {
    if (erased.contains(op))
      continue;
    if (!op->getUses().empty())
      return op->emitError("uses of the op remained live after the pass");
    deleteOp(op);
  }
  for (Block *block : llvm::reverse(blockWorklist)) {
    if (!erased.insert(block).second)
      continue;
    if (!block->getUses().empty())
      return mlir::emitError(loc,
                             "uses of the block remained live after the pass");
    block->dropAllDefinedValueUses();
    block->erase();
  }
  return success();
}

void IRManager::deleteOp(Operation *op) {
  if (!erased.insert(op).second)
    return;
  for (auto &region : op->getRegions()) {
    for (auto &block : llvm::make_early_inc_range(region)) {
      for (auto &op : llvm::make_early_inc_range(block))
        deleteOp(&op);
      if (!erased.insert(&block).second)
        continue;
      block.dropAllDefinedValueUses();
      block.erase();
    }
  }
  op->dropAllReferences();
  op->dropAllUses();
  op->erase();
}

void IRManager::eraseOp(Operation *op) {
  if (op->getUses().empty())
    deleteOp(op);
  else
    opWorklist.push_back(op);
}

void IRManager::eraseBlock(Block *block) { blockWorklist.push_back(block); }

//===----------------------------------------------------------------------===//
// CGDriverImpl
//===----------------------------------------------------------------------===//
CGDriverImpl::CGDriverImpl(CGDriver *driver, const FrozenPatternSet &patterns)
    : applicator(patterns), driver(driver) {
  applicator.applyDefaultCostModel();
}

void CGDriverImpl::logResult(StringRef result, const llvm::Twine &msg) {
  logger.unindent();
  logger.startLine() << "} -> " << result;
  if (!msg.isTriviallyEmpty())
    logger.getOStream() << " : " << msg;
  logger.getOStream() << "\n";
}

const CGPattern *CGDriverImpl::getPattern(Operation *op) {
#ifndef NDEBUG
  auto onFailure = [&](const Pattern &pattern) {
    LLVM_DEBUG({
      logger.getOStream() << "\n";
      logger.startLine() << "* Pattern " << pattern.getDebugName() << " : '"
                         << op->getName() << " -> (";
      llvm::interleaveComma(pattern.getGeneratedOps(), logger.getOStream());
      logger.getOStream() << ")' {\n";
      logger.indent();
      logResult("failure", "pattern failed to match");
    });
  };
#else
  function_ref<void(const Pattern &)> onFailure = {};
#endif
  return applicator.getPattern<CGPattern>(op, {}, onFailure);
}

CGResult CGDriverImpl::genOp(Operation *op) {
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
    LLVM_DEBUG(logResult("success", "pattern matched successfully"));
  };
#endif
  auto [cachedResult, found] = driver->lookup(op);
  if (found)
    return cachedResult;
  CGResult result = nullptr;
  if (const CGPattern *pattern = getPattern(op)) {
#ifndef NDEBUG
    canApply(*pattern);
#endif
    driver->setInsertionPoint(op);
    result = pattern->generate(op, *driver);
    driver->map(op, result);
//    if (rootOp)
//      LLVM_DEBUG(rootOp->print(llvm::dbgs()););
#ifndef NDEBUG
    onSuccess(*pattern);
#endif
  }
  return result;
}

void CGDriverImpl::addToWorklist(Operation *op) { worklist.push_front(op); }

LogicalResult CGDriverImpl::processWorklist() {
  while (worklist.size() > 0) {
    genOp(worklist.pop().get());
  }
  return success();
}

LogicalResult CGDriverImpl::run(Operation *op) {
  rootOp = op;
  for (mlir::Region &region : llvm::make_early_inc_range(op->getRegions()))
    for (mlir::Block &block : llvm::make_early_inc_range(region))
      for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
        addToWorklist(&op);
        if (failed(processWorklist()))
          return failure();
      }
  return irManager.cleanup(op->getLoc());
}

void CGDriverImpl::notifyOperationModified(Operation *op) {}

void CGDriverImpl::notifyOperationInserted(Operation *op,
                                           OpBuilder::InsertPoint previous) {
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  addToWorklist(op);
}

void CGDriverImpl::notifyOperationErased(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Erase  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  irManager.eraseOp(op);
}

void CGDriverImpl::notifyOperationReplaced(Operation *op,
                                           ValueRange replacement) {
  LLVM_DEBUG({
    logger.startLine() << "** Replace : '" << op->getName() << "'(" << op
                       << ")\n";
  });
}

void CGDriverImpl::notifyBlockInserted(Block *block, Region *previous,
                                       Region::iterator previousIt) {
  for (auto &op : block->getOperations()) {
    notifyOperationInserted(&op, OpBuilder::InsertPoint{});
  }
}

void CGDriverImpl::notifyBlockErased(Block *block) {
  irManager.eraseBlock(block);
}

//===----------------------------------------------------------------------===//
// CodegenRewriter
//===----------------------------------------------------------------------===//

CGDriver::CGDriver(XBContext *context, const FrozenPatternSet &patterns,
                   const mlir::TypeConverter *typeConverter)
    : PatternRewriter(context->getMLIRContext()), context(context),
      typeConverter(typeConverter), castInfo(typeConverter) {
  impl.reset(new CGDriverImpl(this, patterns));
  setListener(impl.get());
}

CGResult CGDriver::genOp(Operation *op) { return impl->genOp(op); }

void CGDriver::eraseBlock(Block *block) {
  for (auto &op : llvm::make_early_inc_range(*block))
    eraseOp(&op);
  // Notify the listener.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyBlockErased(block);
}

void CGDriver::eraseOp(Operation *op) {
  // Notify the listener.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationErased(op);
}

void CGDriver::replaceOp(Operation *op, Operation *newOp) {
  assert(op && newOp && "expected non-null op");
  // Notify the listener.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationReplaced(op, newOp);
  eraseOp(op);
}

void CGDriver::replaceOp(Operation *op, ValueRange newValues) {
  assert(op && "expected non-null op");
  // Notify the listener.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationReplaced(op, newValues);
  eraseOp(op);
}

Type CGCastInfo::convertType(Type type) const {
  assert(typeConverter && "invalid type converter");
  return typeConverter->convertType(type);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(xblang::codegen::CGCastInfo);

//===----------------------------------------------------------------------===//
// CodegenRewriter
//===----------------------------------------------------------------------===//

Type CGPattern::convertType(Type type) const {
  assert(typeConverter && "invalid type converter");
  return typeConverter->convertType(type);
}

//===----------------------------------------------------------------------===//
// Driver functions
//===----------------------------------------------------------------------===//

LogicalResult
xblang::codegen::applyCodegenDriver(mlir::Operation *op,
                                    const FrozenPatternSet &patterns,
                                    const mlir::TypeConverter *typeConverter) {
  XBContextDialect *dialect =
      op->getContext()->getLoadedDialect<XBContextDialect>();
  if (!dialect)
    return failure();
  CGDriver driver(&dialect->getContext(), patterns, typeConverter);
  driver->setPrimitiveCast(isPrimitiveCast, createCastOp);
  if (failed(driver.getImpl()->run(op))) {
    LLVM_DEBUG({ op->print(llvm::dbgs()); });
    return failure();
  }
  return success();
}
