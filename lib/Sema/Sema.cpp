//===- Sema.cpp - Semantic checker driver ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the semantic checker driver.
//
//===----------------------------------------------------------------------===//

#include "xblang/Sema/Sema.h"
#include "mlir/IR/PatternMatch.h"
#include "xblang/Basic/PatternApplicator.h"
#include "xblang/Eval/Eval.h"
#include "xblang/Interfaces/SymbolTable.h"
#include "xblang/Support/Format.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include <random>

#define DEBUG_TYPE "sema"

using namespace xblang;
using namespace xblang::sema;
using namespace xblang::sema::detail;

namespace xblang {
namespace sema {
struct SemaState : public IRWorklistElementStatus {
  typedef enum : uint32_t {
    Uninitialized = 0,
    None,
    InChecks,
    // Keep this order for operator bool
    Succeeded,
    Failed,
    Rescheduled,
    Deferred,
  } InternalState;

  SemaState &operator=(InternalState is) {
    state = is;
    return *this;
  }

  SemaState &operator++() {
    IRWorklistElementStatus::operator++();
    return *this;
  }

  bool operator==(InternalState state) { return this->state == state; }

  // Returns true if the state is fully determined.
  operator bool() const { return Succeeded <= state && state <= Deferred; }

  operator SemaResult() const {
    switch (state) {
    case Succeeded:
      return SemaResult::success();
    case Rescheduled:
      return SemaResult::reschedule();
    case Deferred:
      return SemaResult::defer();
    default:
      return SemaResult::failure();
    }
  }

  // Returns the number of times the element has been visited.
  uint32_t &getCount() { return visitCount; }

  const SemaPattern *pattern = {};
};
} // namespace sema
} // namespace xblang

namespace {
struct RandomSalt {
  RandomSalt() : device(), generator(device()), distribution(0, 62) {}

  // Generates a random characher in [a-zA-Z0-9]
  char getChar() {
    int c = distribution(generator);
    if (c < 26) // Return a-z in ASCII
      return c + 97;
    else if (c < 52) // Return A-Z in ASCII
      return (c - 26) + 65;
    else if (c < 62) // Return 0-9 in ASCII
      return (c - 52) + 48;
    return '_';
  }

  // Generates a random string.
  SmallString<16> operator()(size_t sz = 6) {
    SmallString<16> salt;
    salt.reserve(sz);
    for (size_t i = 0; i < sz; ++i)
      salt.push_back(getChar());
    return salt;
  }

  std::random_device device;
  std::mt19937 generator;
  std::uniform_int_distribution<> distribution;
};
} // namespace

namespace xblang {
namespace sema {
namespace detail {
struct SemaDriverImpl : public mlir::RewriterBase::Listener {
  template <typename T>
  using Queue = std::deque<T>;
  using Worklist = GenericIRWorklist<Queue, SemaWorkListElement>;
  SemaDriverImpl(SemaDriver *driver, const FrozenPatternSet &set,
                 SymbolTableContext &symTables);
  /// Returns an unique symbol identifier.
  StringAttr getSymbolUSR(StringAttr sym_id);
  // Returns the semantic for a given operation.
  const SemaPattern *getPattern(mlir::Operation *op);
  /// Checks an operation.
  SemaResult checkOp(SemaWorkListElement &elem, bool forceCheck);
  SemaResult checkOp(mlir::Operation *op, SymbolTable *symTable,
                     bool forceCheck);
  /// Checks the operands of an operation.
  SemaResult checkOperands(mlir::Operation *op, SymbolTable *symTable);
  /// Checks the regions of an operation.
  SemaResult checkRegions(mlir::Operation *op, SymbolTable *symTable);
  /// Schedules an operation for verification.
  SemaResult processWorklist();
  // Runs the driver.
  SemaResult run(mlir::Operation *op);
  /// Applies a semantic checker pattern.
  SemaResult applyPattern(const SemaPattern *pattern,
                          SemaWorkListElement element);

  /// Notify the driver that the specified operation was inserted. Update the
  /// worklist as needed: The operation is enqueued depending on scope and
  /// strict mode.
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;

private:
  friend class xblang::sema::SemaDriver;
  // Pointer to the main driver.
  SemaDriver *driver;
  // Mapping between operation type IDs and semantic patterns.
  GenericPatternApplicator applicator;
  // Driver worklist.
  Worklist worklist{};
  // Sema driver cache.
  DenseMap<mlir::Operation *, SemaState> resultCache{};
  // List of deferred nodes.
  SmallVector<mlir::Operation *> deferredNodes{};
  // Symbol USR map.
  DenseSet<StringAttr> usrMap{};
  // Symbol table context.
  SymbolTableContext &symTables;
  // Random salt generator.
  RandomSalt salt{};
  // Logger.
  llvm::ScopedPrinter logger{llvm::dbgs()};
  // Logs a result to the logger.
  void logResult(StringRef result, const llvm::Twine &msg = {});
};
} // namespace detail
} // namespace sema
} // namespace xblang

//===----------------------------------------------------------------------===//
// SemaDriverImpl
//===----------------------------------------------------------------------===//

SemaDriverImpl::SemaDriverImpl(SemaDriver *driver, const FrozenPatternSet &set,
                               SymbolTableContext &symTables)
    : driver(driver), applicator(set), symTables(symTables) {
  applicator.applyDefaultCostModel();
}

void SemaDriverImpl::logResult(StringRef result, const llvm::Twine &msg) {
  logger.unindent();
  logger.startLine() << "} -> " << result;
  if (!msg.isTriviallyEmpty())
    logger.getOStream() << " : " << msg;
  logger.getOStream() << "\n";
}

const SemaPattern *SemaDriverImpl::getPattern(mlir::Operation *op) {
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
  return applicator.getPattern<SemaPattern>(op, {}, onFailure);
}

SemaResult SemaDriverImpl::checkOperands(mlir::Operation *op,
                                         SymbolTable *symTable) {
  SemaResult res = LogicalResult::success();
  LLVM_DEBUG({
    logger.startLine() << "Op: checking operands with table (" << symTable
                       << ")\n";
  });
  for (mlir::Value operand : llvm::make_early_inc_range(op->getOperands())) {
    if (auto result = driver->checkValue(operand, symTable);
        !result.succeeded()) {
      res = result.rescheduled() ? driver->defer() : result;
      break;
    }
  }
  LLVM_DEBUG({
    logger.startLine() << "Op: finished checking operands with table ("
                       << symTable << ")\n";
  });
  return res;
}

SemaResult SemaDriverImpl::checkRegions(mlir::Operation *op,
                                        SymbolTable *symTable) {
  if (isa<SymbolTableInterface>(op) ||
      op->hasTrait<mlir::OpTrait::SymbolTable>()) {
    symTable = symTables.get(op);
    assert(symTable && "symbol table is missing");
    if (!symTable)
      return LogicalResult::failure();
    if (symTable->getKind() == SymbolTableKind::Ordered)
      symTable->clear();
  }
  LLVM_DEBUG({
    logger.startLine() << "Op: checking regions with table (" << symTable
                       << ")\n";
  });
  SemaResult res = LogicalResult::success();
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region) {
      for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
        if (auto result = checkOp(&op, symTable, false); !result.succeeded()) {
          res = result.rescheduled() ? driver->defer() : result;
          break;
        }
      }
    }
  }
  LLVM_DEBUG({
    logger.startLine() << "Op: finished checking regions with table ("
                       << symTable << ")\n";
  });
  return res;
}

SemaResult SemaDriverImpl::applyPattern(const SemaPattern *pattern,
                                        SemaWorkListElement element) {
  SemaResult result = success();
  assert(pattern && "invalid pattern");
#ifndef NDEBUG
  auto canApply = [&](const Pattern &pattern) {
    LLVM_DEBUG({
      logger.startLine() << "* Pattern " << pattern.getDebugName() << " : '"
                         << element.get()->getName() << "[" << element.get()
                         << "]"
                         << " -> (";
      llvm::interleaveComma(pattern.getGeneratedOps(), logger.getOStream());
      logger.getOStream() << ")' {\n";
      logger.indent();
    });
    return true;
  };
  auto postApply = [&](const Pattern &pattern) {
    LLVM_DEBUG(logResult("pattern matched successfully",
                         "semantic result: `" + result.toString() + "`"));
  };
#endif
#ifndef NDEBUG
  canApply(*pattern);
#endif
  auto grd = driver->guard(*driver, element.get());
  result = pattern->check(element, *driver);
#ifndef NDEBUG
  postApply(*pattern);
#endif
  return result;
}

SemaResult SemaDriverImpl::checkOp(SemaWorkListElement &elem, bool forceCheck) {
  mlir::Operation *op = elem.get();
  SymbolTable *symTable = elem.getSymbolTable();
  SemaState &cache = resultCache[op];
  // Return the cache if it's not empty.
  if (cache == SemaState::Failed || (cache && !forceCheck))
    return cache;
  // Detect cycles in the semantic checks.
  if (cache == SemaState::InChecks)
    return op->emitError(
        "operation is not verifiable, semantic cycle detected");
  // Initialize the cache if it's the first time seeing the op.
  if (cache == SemaState::Uninitialized) {
    cache.pattern = getPattern(op);
    cache = SemaState::None;
  }
  SemaState cacheTmp = cache;
  if (forceCheck)
    cache = SemaState::None;
  bool isDeferred = cache == SemaState::Deferred;
  // Return if there is nothing to check.
  if (!cache.pattern)
    return mlir::success();
  // Get the symbol table if none was provided.
  if (!symTable)
    if (mlir::Operation *symOp = op->getParentOfType<SymbolTableInterface>())
      symTable = symTables.get(symOp);
  // Mark the operation as being checked to detect cycles.
  cache = SemaState::InChecks;
  // Append a potential symbol if the table is ordered.
  if (symTable && symTable->getKind() == SymbolTableKind::Ordered)
    if (auto sym = dyn_cast<Symbol>(op); sym && symTable != symTables.get(op))
      symTable->insert(sym);
  // Check the operation.
  SemaResult result =
      applyPattern(cache.pattern, SemaWorkListElement(op, symTable, cacheTmp));
  // Update the cache.
  {
    SemaState &cache = resultCache[op];
    if (result.succeeded())
      ++cache = SemaState::Succeeded;
    else if (result.failed())
      ++cache = SemaState::Failed;
    else if (result.rescheduled())
      ++cache = SemaState::Rescheduled;
    else // If it was deferred, don't increment the visitor counter.
      cache = SemaState::Deferred;
    elem = cache;
    if (result.requiresReschedule())
      worklist.push_front(elem);
  }
  // Fail if the op has been deferred many times.
  if (isDeferred && result.deferred()) {
    return mlir::failure();
  }
  return result.rescheduled() ? driver->defer() : result;
}

SemaResult SemaDriverImpl::checkOp(mlir::Operation *op, SymbolTable *symTable,
                                   bool forceCheck) {
  SemaWorkListElement elem(op, symTable);
  return checkOp(elem, forceCheck);
}

void SemaDriverImpl::notifyOperationInserted(Operation *op,
                                             OpBuilder::InsertPoint previous) {
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  worklist.push_front(SemaWorkListElement(op, nullptr));
}

SemaResult SemaDriverImpl::processWorklist() {
  size_t errorCount = 0;
  while (worklist.size() > 0) {
    SemaWorkListElement elem = worklist.pop(false);
    // Detect if an element has been rescheduled too many times.
    if (elem.getCount() > 10) {
      elem.get()->emitError(
          "max number of reschedules reached, aborting semantic checks");
      return mlir::failure();
    }
    // Check the op.
    SemaResult result = checkOp(elem, true);
    if (result.failed())
      ++errorCount;
    // Abort after 10 errors.
    if (errorCount > 10)
      return mlir::failure();
  }
  return errorCount == 0 ? mlir::success() : mlir::failure();
}

SemaResult SemaDriverImpl::run(mlir::Operation *op) {
  worklist.push_back(SemaWorkListElement(op, nullptr));
  return processWorklist();
}

StringAttr SemaDriverImpl::getSymbolUSR(StringAttr sym_id) {
  StringAttr usr;
  size_t i = 0, max_its = 1000;
  while (i++ < max_its) {
    auto str = salt(4);
    usr = driver->getStringAttr(sym_id.getValue() + "_" + str.str());
    if (usrMap.insert(usr).second)
      break;
  }
  assert(i < max_its && "max its reached");
  return usr;
}

//===----------------------------------------------------------------------===//
// SemaResult
//===----------------------------------------------------------------------===//

llvm::StringRef SemaResult::toString() const {
  switch (result) {
  case Success:
    return "succeeded";
  case Failure:
    return "failed";
  case SuccessAndReschedule:
    return "succeeded and rescheduled";
  case Reschedule:
    return "rescheduled";
  case Defer:
    return "deferred";
  }
  return "";
}

//===----------------------------------------------------------------------===//
// SemaDriver
//===----------------------------------------------------------------------===//

SemaDriver::SemaDriver(XBContext &context, const FrozenPatternSet &patterns,
                       SymbolTableContext &symTables, TypeSystem &typeSystem,
                       eval::EvalDriver *evaluateDriver)
    : PatternRewriter(context.getMLIRContext()),
      impl(new SemaDriverImpl(this, patterns, symTables)), context(&context),
      typeSystem(&typeSystem), evaluateDriver(evaluateDriver) {
  setListener(impl.get());
}

SemaDriver::~SemaDriver() = default;

SemaResult SemaDriver::checkOperands(mlir::Operation *op,
                                     SymbolTable *symTable) {
  return impl->checkOperands(op, symTable);
}

SemaResult SemaDriver::checkRegions(mlir::Operation *op,
                                    SymbolTable *symTable) {
  return impl->checkRegions(op, symTable);
}

SemaResult SemaDriver::checkOp(mlir::Operation *op, SymbolTable *symTable,
                               bool forceCheck) {
  return impl->checkOp(op, symTable, forceCheck);
}

StringAttr SemaDriver::getSymbolUSR(StringAttr sym_id) {
  return impl->getSymbolUSR(sym_id);
}

Operation *SemaDriver::lookupUSR(Attribute usr) {
  return impl->symTables.lookupUSR(usr);
}

void SemaDriver::setUSR(Attribute usr, Operation *op) {
  impl->symTables.setUSR(usr, op);
}

Attribute SemaDriver::eval(mlir::Operation *op, ArrayRef<Attribute> args) {
  return evaluateDriver ? evaluateDriver->eval(op, args) : nullptr;
}

LogicalResult SemaDriver::buildTables(Operation *op, SymbolTable *parent) {
  return impl->symTables.buildTables(op, parent);
}

//===----------------------------------------------------------------------===//
// Sema API
//===----------------------------------------------------------------------===//

LogicalResult xblang::sema::applySemaDriver(mlir::Operation *op,
                                            const FrozenPatternSet &set,
                                            TypeSystem &typeSystem,
                                            eval::EvalDriver *evaluateDriver) {
  auto symTables = SymbolTableContext::create(op);
  if (mlir::failed(symTables))
    return mlir::failure();
  SemaDriver driver(*set.getContext(), set, *symTables, typeSystem,
                    evaluateDriver);
  if (!driver.getImpl()->run(op).succeeded())
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Type utilities
//===----------------------------------------------------------------------===//

Type xblang::promotePrimitiveTypes(Type lhsTy, Type rhsTy) {
  // Return early if the types are the same or one is null.
  if (lhsTy == rhsTy)
    return lhsTy;
  if (!lhsTy || !rhsTy)
    return nullptr;
  if (!isPrimitiveType(lhsTy) || !isPrimitiveType(rhsTy))
    return nullptr;
  // Promote between integer types.
  auto lhsInt = dyn_cast<IntegerType>(lhsTy);
  auto rhsInt = dyn_cast<IntegerType>(rhsTy);
  if (lhsInt && rhsInt) {
    unsigned w1 = lhsInt.getWidth();
    unsigned w2 = rhsInt.getWidth();
    if (w1 > w2) {
      return lhsTy;
    } else if (w1 < w2) {
      return rhsTy;
    } else {
      if (lhsInt.isUnsigned())
        return lhsTy;
      else
        return rhsTy;
    }
  }
  // Promote between float types.
  auto lhsFP = dyn_cast<FloatType>(lhsTy);
  auto rhsFP = dyn_cast<FloatType>(rhsTy);
  if (lhsFP && rhsFP)
    return lhsFP.getWidth() > rhsFP.getWidth() ? lhsTy : rhsTy;
  // Promote int to floats.
  if (lhsInt && rhsFP)
    return rhsTy;
  if (rhsInt && lhsFP)
    return lhsTy;
  // Promote int to index.
  auto lhsIndex = dyn_cast<IndexType>(lhsTy);
  auto rhsIndex = dyn_cast<IndexType>(rhsTy);
  if (lhsInt && rhsIndex)
    return rhsTy;
  if (rhsInt && lhsIndex)
    return lhsTy;
  // Promote index to float.
  if (lhsIndex && rhsFP)
    return rhsTy;
  if (rhsIndex && lhsFP)
    return lhsTy;
  return nullptr;
}
