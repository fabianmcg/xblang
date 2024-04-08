//===- Sema.h - Sema driver  -------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the semantic checker driver.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SEMA_SEMA_H
#define XBLANG_SEMA_SEMA_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "xblang/Basic/Context.h"
#include "xblang/Basic/Pattern.h"
#include "xblang/Basic/TypeSystem.h"
#include "xblang/Sema/TypeUtil.h"
#include "xblang/Support/BuilderBase.h"
#include "xblang/Support/Worklist.h"
#include "xblang/XLG/IR/XLGTypes.h"
#include "llvm/ADT/DenseMap.h"

namespace xblang {
class SymbolTable;
class SymbolTableContext;

namespace eval {
class EvalDriver;
}

namespace sema {
class SemaDriver;

//===----------------------------------------------------------------------===//
// Semantic result
//===----------------------------------------------------------------------===//
class SemaResult {
public:
  typedef enum : uint8_t {
    Success,              // The operation is semantically valid.
    Failure,              // The operation is semantically invalid.
    SuccessAndReschedule, // The semantics checks have to be rescheduled but the
                          // op reports as successful.
    Reschedule,           // The semantics checks have to be rescheduled.
    Defer,                // The semantics checks have been deferred.
  } Result;

  SemaResult(mlir::LogicalResult result)
      : result(mlir::succeeded(result) ? Success : Failure) {}

  SemaResult(mlir::InFlightDiagnostic result)
      : result(result.operator mlir::LogicalResult().succeeded() ? Success
                                                                 : Failure) {}

  operator bool() const { return !failed(); }

  /// Creates a successful semantic result.
  static SemaResult success() { return SemaResult(Success); }

  /// Creates a deferred semantic result.
  static SemaResult failure() { return SemaResult(Failure); }

  /// Creates a quiet reschedule op semantic result.
  static SemaResult successAndReschedule() {
    return SemaResult(SuccessAndReschedule);
  }

  /// Creates a reschedule op semantic result.
  static SemaResult reschedule() { return SemaResult(Reschedule); }

  /// Returns true if the semantic result is successful.
  bool succeeded() const {
    return result == Success || result == SuccessAndReschedule;
  }

  /// Returns true if the semantic result is failure.
  bool failed() const { return result == Failure; }

  /// Returns true if the semantic result is undetermined.
  bool rescheduled() const { return result == Reschedule; }

  /// Returns true if the semantic result is failure.
  bool deferred() const { return result == Defer; }

  /// Returns true if the op has to be rescheduled.
  bool requiresReschedule() const {
    return result == Reschedule || result == Defer ||
           result == SuccessAndReschedule;
  }

  /// Returns a string representation of the result.
  llvm::StringRef toString() const;

private:
  friend class SemaDriver;
  friend struct SemaState;

  /// Creates a reschedule op semantic result.
  static SemaResult defer() { return SemaResult(Defer); }

  SemaResult(Result result) : result(result) {}

  Result result;
};

//===----------------------------------------------------------------------===//
// Semantic driver
//===----------------------------------------------------------------------===//
namespace detail {
struct SemaDriverImpl;
}

/// Semantic checker driver. This class must never be sub-classed or
/// instantiated; use the create method to obtain a driver instance.
class SemaDriver : public mlir::PatternRewriter, public ::xblang::BuilderBase {
public:
  SemaDriver(XBContext &context, const FrozenPatternSet &patterns,
             SymbolTableContext &symTables, TypeSystem &typeSystem,
             eval::EvalDriver *evaluateDriver);

  ~SemaDriver();

  /// Returns the XBLang context
  XBContext *getXBContext() const { return context; }

  /// Checks an operation.
  SemaResult checkOp(mlir::Operation *op, SymbolTable *symTable,
                     bool forceCheck = false);
  /// Checks the operands of an operation.
  SemaResult checkOperands(mlir::Operation *op, SymbolTable *symTable);
  /// Checks the regions of an operation.
  SemaResult checkRegions(mlir::Operation *op, SymbolTable *symTable);

  /// Checks a value.
  SemaResult checkValue(mlir::Value value, SymbolTable *symTable,
                        bool forceCheck = false) {
    if (value)
      if (auto op = value.getDefiningOp())
        return checkOp(op, symTable, forceCheck);
    return mlir::success();
  }

  /// Returns the driver implementation.
  detail::SemaDriverImpl *getImpl() { return impl.get(); }

  /// Returns a concept interface for the given operation.
  template <typename Interface>
  Interface getInterface(mlir::Operation *op) {
    return Interface::get(getXBContext(), op);
  }

  /// Returns the concept stored in the context.
  template <typename Concept>
  Concept *getConcept() {
    return Concept::get(getXBContext());
  }

  /// Returns a class type with the specified concept.
  xlg::ConceptType getConceptClass(Concept *cncpt) {
    return xlg::ConceptType::get(getContext(), cncpt);
  }

  template <typename Concept>
  xlg::ConceptType getConceptClass() {
    return xlg::ConceptType::get(getContext(), getConcept<Concept>());
  }

  /// Returns an unique symbol identifier.
  StringAttr getSymbolUSR(StringAttr sym_id);

  /// Looks up an operation by the USR.
  Operation *lookupUSR(Attribute usr);

  /// Sets the USR for an operation.
  void setUSR(Attribute usr, Operation *op);

  /// Allow access to the type system.
  TypeSystem *operator->() { return typeSystem; }

  const TypeSystem *operator->() const { return typeSystem; }

  /// Invoke the evaluate driver.
  Attribute eval(Operation *op, ArrayRef<Attribute> args = {});

  /// Recursively builds the symbol tables for a new operation and its regions.
  LogicalResult buildTables(Operation *op, SymbolTable *parent);

protected:
  static SemaResult defer() { return SemaResult::defer(); }

  friend struct detail::SemaDriverImpl;
  std::unique_ptr<detail::SemaDriverImpl> impl;
  /// XB context
  XBContext *context;
  /// Type system to use during semantic checks.
  TypeSystem *typeSystem{};
  /// Evaluation driver.
  eval::EvalDriver *evaluateDriver{};
};

//===----------------------------------------------------------------------===//
// Semantic patterns
//===----------------------------------------------------------------------===//

class SemaWorkListElement : public IRWorklistElement {
public:
  SemaWorkListElement() = default;

  SemaWorkListElement(mlir::Operation *op, SymbolTable *symTable,
                      IRWorklistElementStatus status = {})
      : IRWorklistElement(op, status), symTable(symTable) {}

  SemaWorkListElement &operator=(IRWorklistElementStatus status) {
    IRWorklistElementStatus::operator=(status);
    return *this;
  }

  /// Increments the visitor counter.
  SemaWorkListElement &operator++() {
    IRWorklistElement::operator++();
    return *this;
  }

  /// Returns the symbol table.
  SymbolTable *getSymbolTable() const { return symTable; }

private:
  SymbolTable *symTable = {};
};

//===----------------------------------------------------------------------===//
// Semantic patterns
//===----------------------------------------------------------------------===//

/// Semantic checker pattern. The driver will call the check and check
/// methods and return a semantic result.
class SemaPattern : public GenericPattern {
protected:
  using GenericPattern::GenericPattern;

  template <typename TagOrStringRef>
  SemaPattern(TagOrStringRef arg, PatternBenefit benefit, MLIRContext *context,
              ArrayRef<StringRef> generatedNames = {})
      : GenericPattern(arg, benefit, context, generatedNames) {}

  template <typename Tag>
  SemaPattern(Tag tag, TypeID typeID, PatternBenefit benefit,
              MLIRContext *context, ArrayRef<StringRef> generatedNames = {})
      : GenericPattern(tag, typeID, benefit, context, generatedNames) {}

public:
  /// Checks the operands, the regions and the operation.
  virtual SemaResult check(SemaWorkListElement op, SemaDriver &driver) const {
    // Check the op operands.
    if (SemaResult result = driver.checkOperands(op.get(), op.getSymbolTable());
        !result.succeeded())
      return result;

    // Check the op regions.
    if (SemaResult result = driver.checkRegions(op.get(), op.getSymbolTable());
        !result.succeeded())
      return result;

    // Check the op.
    return checkOp(op, driver);
  }

  /// Checks the operation.
  virtual SemaResult checkOp(SemaWorkListElement op, SemaDriver &driver) const {
    return mlir::success();
  }
};

/// SemaOpPattern is a wrapper around SemaPattern that allows checking against
/// an instance of a derived operation class.
template <typename SourceOp>
class SemaOpPattern : public SemaPattern {
public:
  using Base = SemaOpPattern;
  using Op = SourceOp;
  using Status = IRWorklistElementStatus;

  SemaOpPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : SemaPattern(Op::getOperationName(), benefit, context) {}

  /// Wrappers around the SemaPattern methods that pass the derived op
  /// type.
  LogicalResult match(Operation *op) const final { return match(cast<Op>(op)); }

  SemaResult check(SemaWorkListElement op, SemaDriver &driver) const final {
    return check(mlir::cast<Op>(op.get()), op, op.getSymbolTable(), driver);
  }

  SemaResult checkOp(SemaWorkListElement op, SemaDriver &driver) const final {
    return checkOp(mlir::cast<Op>(op.get()), op, op.getSymbolTable(), driver);
  }

  /// SemaPattern concrete op methods.
  virtual LogicalResult match(Op op) const { return success(); }

  virtual SemaResult check(Op op, Status status, SymbolTable *symTable,
                           SemaDriver &driver) const {
    // Check the op operands.
    if (SemaResult result = driver.checkOperands(op, symTable);
        !result.succeeded())
      return result;

    // Check the op regions.
    if (SemaResult result = driver.checkRegions(op, symTable);
        !result.succeeded())
      return result;

    // Check the op.
    return checkOp(op, status, symTable, driver);
  }

  virtual SemaResult checkOp(Op op, Status status, SymbolTable *symTable,
                             SemaDriver &driver) const {
    return mlir::success();
  }
};

/// SemaOpPattern is a wrapper around SemaPattern that allows checking against
/// an instance of an interface.
template <typename Iface>
class InterfaceSemaPattern : public SemaPattern {
public:
  using Base = InterfaceSemaPattern;
  using Interface = Iface;
  using Status = IRWorklistElementStatus;

  InterfaceSemaPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : SemaPattern(Pattern::MatchInterfaceOpTypeTag(),
                    TypeID::get<Interface>(), benefit, context) {}

  /// Wrappers around the SemaPattern methods that pass the derived interface
  /// type.
  LogicalResult match(Operation *iface) const final {
    return match(cast<Interface>(iface));
  }

  SemaResult check(SemaWorkListElement iface, SemaDriver &driver) const final {
    return check(mlir::cast<Interface>(iface.get()), iface,
                 iface.getSymbolTable(), driver);
  }

  SemaResult checkOp(SemaWorkListElement iface,
                     SemaDriver &driver) const final {
    return checkOp(mlir::cast<Interface>(iface.get()), iface,
                   iface.getSymbolTable(), driver);
  }

  /// SemaPattern concrete iface methods.
  virtual LogicalResult match(Interface iface) const { return success(); }

  virtual SemaResult check(Interface iface, Status status,
                           SymbolTable *symTable, SemaDriver &driver) const {
    // Check the iface operands.
    if (SemaResult result = driver.checkOperands(iface, symTable);
        !result.succeeded())
      return result;

    // Check the iface regions.
    if (SemaResult result = driver.checkRegions(iface, symTable);
        !result.succeeded())
      return result;

    // Check the iface.
    return checkOp(iface, status, symTable, driver);
  }

  virtual SemaResult checkOp(Interface iface, Status status,
                             SymbolTable *symTable, SemaDriver &driver) const {
    return mlir::success();
  }
};

mlir::LogicalResult applySemaDriver(mlir::Operation *op,
                                    const FrozenPatternSet &set,
                                    TypeSystem &typeSystem,
                                    eval::EvalDriver *evaluateDriver = nullptr);
} // namespace sema
} // namespace xblang

#endif // XBLANG_SEMA_SEMA_H
