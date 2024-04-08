//===- Eval.h - Constant evaluation driver -----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the constant evaluation driver.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_EVAL_EVAL_H
#define XBLANG_EVAL_EVAL_H

#include "mlir/IR/PatternMatch.h"
#include "xblang/Basic/Pattern.h"
#include "xblang/Support/LLVM.h"

namespace xblang {
namespace eval {
namespace detail {
class EvalDriverImpl;
}
class EvalDriver;
//===----------------------------------------------------------------------===//
// Eval API
//===----------------------------------------------------------------------===//
/// Applies the evaluate driver to an operation.
Attribute evaluate(Operation *op, ArrayRef<Attribute> args,
                   const FrozenPatternSet &patterns);

//===----------------------------------------------------------------------===//
// Eval Patterns
//===----------------------------------------------------------------------===//
class EvalDriver : public mlir::Builder {
public:
  EvalDriver(XBContext *context, const FrozenPatternSet &patterns);
  ~EvalDriver();

  /// Evaluates an operation.
  Attribute eval(Operation *op, ArrayRef<Attribute> args = {});

  Attribute eval(Value val) {
    assert(val && "invalid value");
    if (auto op = val.getDefiningOp())
      return eval(op);
    return nullptr;
  }

  /// Returns the XB context.
  XBContext *getXBContext() const { return context; }

  /// Returns a concept interface for the given operation.
  template <typename Interface>
  Interface getInterface(mlir::Operation *op) {
    return Interface::get(getXBContext(), op);
  }

  /// Returns the driver implementation.
  detail::EvalDriverImpl *getImpl() { return impl.get(); }

private:
  friend class detail::EvalDriverImpl;
  std::unique_ptr<detail::EvalDriverImpl> impl{};
  XBContext *context;
};

//===----------------------------------------------------------------------===//
// Eval Patterns
//===----------------------------------------------------------------------===//

/// Base class for the code generation patterns.
class EvalPattern : public GenericPattern {
protected:
  using GenericPattern::GenericPattern;

public:
  /// Evaluates an op, returning an attribute.
  virtual Attribute eval(mlir::Operation *op, ArrayRef<Attribute> args,
                         EvalDriver &driver) const = 0;
};

/// Class for evaluation patterns based on a concrete op.
template <typename SourceOp>
class OpEvalPattern : public EvalPattern {
public:
  using Base = OpEvalPattern;
  using Op = SourceOp;
  using OpAdaptor = typename Op::Adaptor;

  OpEvalPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : EvalPattern(Op::getOperationName(), benefit, context) {}

  /// Wrappers around the EvalPattern methods that pass the derived op
  /// type.
  LogicalResult match(Operation *op) const final { return match(cast<Op>(op)); }

  Attribute eval(mlir::Operation *op, ArrayRef<Attribute> args,
                 EvalDriver &driver) const final {
    return eval(cast<Op>(op), args, driver);
  }

  /// Concrete op specific methods.
  virtual LogicalResult match(Op op) const { return success(); }

  virtual Attribute eval(Op op, ArrayRef<Attribute> args,
                         EvalDriver &driver) const {
    return nullptr;
  }
};

/// Class for evaluation patterns based on an interface.
template <typename Iface>
class InterfaceEvalPattern : public EvalPattern {
public:
  using Base = InterfaceEvalPattern;
  using Interface = Iface;

  InterfaceEvalPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : EvalPattern(Pattern::MatchInterfaceOpTypeTag(),
                    TypeID::get<Interface>(), benefit, context) {}

  /// Wrappers around the EvalPattern methods that pass the derived interface
  /// type.
  LogicalResult match(Operation *op) const final {
    return match(cast<Interface>(op));
  }

  Attribute eval(mlir::Operation *op, ArrayRef<Attribute> args,
                 EvalDriver &driver) const final {
    return eval(cast<Interface>(op), args, driver);
  }

  /// Concrete interface specific methods.
  virtual LogicalResult match(Interface op) const { return success(); }

  virtual Attribute eval(Interface op, ArrayRef<Attribute> args,
                         EvalDriver &driver) const {
    return nullptr;
  }
};
} // namespace eval
} // namespace xblang

#endif // XBLANG_EVAL_EVAL_H
