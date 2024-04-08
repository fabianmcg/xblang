//===- Utils.cpp - Eval op utilities -----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op evaluation utilities.
//
//===----------------------------------------------------------------------===//

#include "xblang/Eval/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace xblang;

namespace {
struct ArithEval {
  mlir::Builder &builder;
  Attribute lhs;
  Attribute rhs;
  Type type;
  Location loc;

  ArithEval(mlir::Builder &builder, Attribute lhs, Attribute rhs, Type type,
            Location loc)
      : builder(builder), lhs(lhs), rhs(rhs), type(type), loc(loc) {}

  /// Evaluates a binary operation.
  Attribute evalBinOp(BinaryOperator op);
  /// Evaluates an add operation.
  Attribute evalAddOp();
  /// Evaluates a sub operation.
  Attribute evalSubOp();
  /// Evaluates a mul operation.
  Attribute evalMulOp();
  /// Evaluates a div operation.
  Attribute evalDivOp();
  /// Evaluates a mod operation.
  Attribute evalModOp();
  /// Evaluates a left shift operation.
  Attribute evalLShiftOp();
  /// Evaluates a right shift operation.
  Attribute evalRShiftOp();
  /// Evaluates a compare equal operation.
  Attribute evalCmpEqOp();
  /// Evaluates a compare not equal operation.
  Attribute evalCmpNeqOp();
  /// Evaluates a compare less than operation.
  Attribute evalCmpLtOp();
  /// Evaluates a compare greater than operation.
  Attribute evalCmpGtOp();
  /// Evaluates a compare less equal operation.
  Attribute evalCmpLeqOp();
  /// Evaluates a compare greater equal operation.
  Attribute evalCmpGeqOp();
  /// Evaluates a compare spaceship operation.
  Attribute evalCmpSpaceshipOp();
  /// Evaluates a binary and operation.
  Attribute evalAndOp();
  /// Evaluates a binary or operation.
  Attribute evalOrOp();
  /// Evaluates a binary xor operation.
  Attribute evalXorOp();
  /// Compares 2 integer attributes
  int32_t compare(IntegerAttr lhs, IntegerAttr rhs, bool isSigned);
};

struct CastEval {
  mlir::Builder &builder;

  CastEval(mlir::Builder &builder) : builder(builder) {}

  /// Casts an attribute to an integer type.
  Attribute castToInt(IntegerType type, Type srcType, Attribute expr);
  /// Casts an attribute to a float type.
  Attribute castToFloat(FloatType type, Type srcType, Attribute expr);
  /// Casts an attribute to an index type.
  Attribute castToIndex(IndexType type, Type srcType, Attribute expr);
  /// Casts an attribute.
  Attribute castValue(Type dstType, Type srcType, Attribute expr);
};
} // namespace

Attribute ArithEval::evalAddOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue() +
                                          cast<IntegerAttr>(rhs).getValue());
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(ty, cast<mlir::FloatAttr>(lhs).getValue() +
                                        cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() +
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalMulOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue() *
                                          cast<IntegerAttr>(rhs).getValue());
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(ty, cast<mlir::FloatAttr>(lhs).getValue() *
                                        cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() *
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalSubOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue() -
                                          cast<IntegerAttr>(rhs).getValue());
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(ty, cast<mlir::FloatAttr>(lhs).getValue() -
                                        cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() -
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalDivOp() {
  if (auto ty = dyn_cast<IntegerType>(type)) {
    if (ty.isSigned())
      return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue().sdiv(
                                            cast<IntegerAttr>(rhs).getValue()));
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue().udiv(
                                          cast<IntegerAttr>(rhs).getValue()));
  } else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(ty, cast<mlir::FloatAttr>(lhs).getValue() /
                                        cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() /
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalModOp() {
  if (auto ty = dyn_cast<IntegerType>(type)) {
    if (ty.isSigned())
      return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue().srem(
                                            cast<IntegerAttr>(rhs).getValue()));
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue().urem(
                                          cast<IntegerAttr>(rhs).getValue()));
  } else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() %
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalLShiftOp() {
  if (auto ty = dyn_cast<IntegerType>(type)) {
    if (ty.isSigned())
      return builder.getIntegerAttr(ty,
                                    cast<IntegerAttr>(lhs).getValue()
                                        << cast<IntegerAttr>(rhs).getValue());
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue()
                                          << cast<IntegerAttr>(rhs).getValue());
  } else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt()
                                << cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalRShiftOp() {
  if (auto ty = dyn_cast<IntegerType>(type)) {
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue().ashr(
                                          cast<IntegerAttr>(rhs).getValue()));
  } else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() >>
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalCmpEqOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getBoolAttr(cast<IntegerAttr>(lhs).getValue() ==
                               cast<IntegerAttr>(rhs).getValue());
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getBoolAttr(cast<mlir::FloatAttr>(lhs).getValue() ==
                               cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getBoolAttr(cast<IntegerAttr>(lhs).getValue() ==
                               cast<IntegerAttr>(rhs).getValue());
  return nullptr;
}

Attribute ArithEval::evalCmpNeqOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getBoolAttr(cast<IntegerAttr>(lhs).getValue() !=
                               cast<IntegerAttr>(rhs).getValue());
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getBoolAttr(cast<mlir::FloatAttr>(lhs).getValue() !=
                               cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getBoolAttr(cast<IntegerAttr>(lhs).getValue() !=
                               cast<IntegerAttr>(rhs).getValue());
  return nullptr;
}

int32_t ArithEval::compare(IntegerAttr lhs, IntegerAttr rhs, bool isSigned) {
  APInt l = lhs.getValue();
  APInt r = rhs.getValue();
  if (isSigned) {
    if (l.slt(r))
      return -1;
    else if (l == r)
      return 0;
    return 1;
  }
  if (l.ult(r))
    return -1;
  else if (l == r)
    return 0;
  return 1;
}

Attribute ArithEval::evalCmpLtOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getBoolAttr(compare(cast<IntegerAttr>(lhs),
                                       cast<IntegerAttr>(rhs),
                                       ty.isSigned()) < 0);
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getBoolAttr(cast<mlir::FloatAttr>(lhs).getValue() <
                               cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getBoolAttr(
        compare(cast<IntegerAttr>(lhs), cast<IntegerAttr>(rhs), false) < 0);
  return nullptr;
}

Attribute ArithEval::evalCmpGtOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getBoolAttr(compare(cast<IntegerAttr>(lhs),
                                       cast<IntegerAttr>(rhs),
                                       ty.isSigned()) > 0);
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getBoolAttr(cast<mlir::FloatAttr>(lhs).getValue() >
                               cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getBoolAttr(
        compare(cast<IntegerAttr>(lhs), cast<IntegerAttr>(rhs), false) > 0);
  return nullptr;
}

Attribute ArithEval::evalCmpLeqOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getBoolAttr(compare(cast<IntegerAttr>(lhs),
                                       cast<IntegerAttr>(rhs),
                                       ty.isSigned()) <= 0);
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getBoolAttr(cast<mlir::FloatAttr>(lhs).getValue() <=
                               cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getBoolAttr(
        compare(cast<IntegerAttr>(lhs), cast<IntegerAttr>(rhs), false) <= 0);
  return nullptr;
}

Attribute ArithEval::evalCmpGeqOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getBoolAttr(compare(cast<IntegerAttr>(lhs),
                                       cast<IntegerAttr>(rhs),
                                       ty.isSigned()) >= 0);
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getBoolAttr(cast<mlir::FloatAttr>(lhs).getValue() >=
                               cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getBoolAttr(
        compare(cast<IntegerAttr>(lhs), cast<IntegerAttr>(rhs), false) >= 0);
  return nullptr;
}

Attribute ArithEval::evalCmpSpaceshipOp() {
  if (auto ty = dyn_cast<IntegerType>(type))
    return builder.getBoolAttr(
        compare(cast<IntegerAttr>(lhs), cast<IntegerAttr>(rhs), ty.isSigned()));
  else if (auto ty = dyn_cast<FloatType>(type))
    return builder.getBoolAttr(cast<mlir::FloatAttr>(lhs).getValue() <
                               cast<mlir::FloatAttr>(rhs).getValue());
  else if (isa<IndexType>(type))
    return builder.getBoolAttr(
        compare(cast<IntegerAttr>(lhs), cast<IntegerAttr>(rhs), false));
  return nullptr;
}

Attribute ArithEval::evalAndOp() {
  if (auto ty = dyn_cast<IntegerType>(type)) {
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue() &
                                          cast<IntegerAttr>(rhs).getValue());
  } else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() &
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalOrOp() {
  if (auto ty = dyn_cast<IntegerType>(type)) {
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue() |
                                          cast<IntegerAttr>(rhs).getValue());
  } else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() |
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalXorOp() {
  if (auto ty = dyn_cast<IntegerType>(type)) {
    return builder.getIntegerAttr(ty, cast<IntegerAttr>(lhs).getValue() ^
                                          cast<IntegerAttr>(rhs).getValue());
  } else if (isa<IndexType>(type))
    return builder.getIndexAttr(cast<IntegerAttr>(lhs).getInt() ^
                                cast<IntegerAttr>(rhs).getInt());
  return nullptr;
}

Attribute ArithEval::evalBinOp(BinaryOperator op) {
  switch (op) {
  case BinaryOperator::Add:
    return evalAddOp();
  case BinaryOperator::Sub:
    return evalSubOp();
  case BinaryOperator::Mul:
    return evalMulOp();
  case BinaryOperator::Div:
    return evalDivOp();
  case BinaryOperator::Mod:
    return evalModOp();
  case BinaryOperator::LShift:
    return evalLShiftOp();
  case BinaryOperator::RShift:
    return evalRShiftOp();
  case BinaryOperator::Equal:
    return evalCmpEqOp();
  case BinaryOperator::NEQ:
    return evalCmpNeqOp();
  case BinaryOperator::Less:
    return evalCmpLtOp();
  case BinaryOperator::Greater:
    return evalCmpGtOp();
  case BinaryOperator::LEQ:
    return evalCmpLeqOp();
  case BinaryOperator::GEQ:
    return evalCmpGeqOp();
  case BinaryOperator::Spaceship:
    return evalCmpSpaceshipOp();
  case BinaryOperator::BinaryAnd:
    return evalAndOp();
  case BinaryOperator::BinaryOr:
    return evalOrOp();
  case BinaryOperator::BinaryXor:
    return evalXorOp();
  default:
    break;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CastEval
//===----------------------------------------------------------------------===//

Attribute CastEval::castToInt(IntegerType tgtType, Type srcType,
                              Attribute expr) {
  using namespace mlir;
  if (tgtType.getWidth() == 1) {
    TypedAttr zero{};
    if (isa<IntegerType>(srcType))
      zero = builder.getIntegerAttr(srcType, 0);
    else if (isa<FloatType>(srcType))
      zero = builder.getFloatAttr(srcType, 0);
    else if (isa<IndexType>(srcType))
      zero = builder.getIndexAttr(0);
    else
      return nullptr;
    return ArithEval(builder, expr, zero, srcType, builder.getUnknownLoc())
        .evalCmpNeqOp();
  }
  if (auto srcTy = dyn_cast<IntegerType>(srcType)) {
    return builder.getIntegerAttr(tgtType, cast<IntegerAttr>(expr).getValue());
  } else if (auto srcTy = dyn_cast<FloatType>(srcType)) {
    return builder.getIntegerAttr(tgtType,
                                  cast<FloatAttr>(expr).getValueAsDouble());
  } else if (auto srcTy = dyn_cast<IndexType>(srcType)) {
    return builder.getIntegerAttr(tgtType, cast<IntegerAttr>(expr).getValue());
  }
  return nullptr;
}

Attribute CastEval::castToFloat(FloatType tgtType, Type srcType,
                                Attribute expr) {
  if (auto srcTy = dyn_cast<IntegerType>(srcType)) {
    return builder.getFloatAttr(
        tgtType,
        cast<IntegerAttr>(expr).getValue().roundToDouble(srcTy.isSigned()));
  } else if (auto srcTy = dyn_cast<FloatType>(srcType)) {
    return builder.getFloatAttr(tgtType,
                                cast<mlir::FloatAttr>(expr).getValue());
  } else if (auto srcTy = dyn_cast<IndexType>(srcType))
    return builder.getFloatAttr(
        tgtType, cast<IntegerAttr>(expr).getValue().roundToDouble(false));
  return nullptr;
}

Attribute CastEval::castToIndex(IndexType tgtType, Type srcType,
                                Attribute expr) {
  if (auto srcTy = dyn_cast<IntegerType>(srcType)) {
    return builder.getIndexAttr(
        cast<IntegerAttr>(expr).getValue().getSExtValue());
  } else if (auto srcTy = dyn_cast<FloatType>(srcType)) {
    return builder.getIndexAttr(
        cast<mlir::FloatAttr>(expr).getValue().convertToDouble());
  } else if (auto srcTy = dyn_cast<IndexType>(srcType)) {
    return expr;
  }
  return nullptr;
}

Attribute CastEval::castValue(Type dstType, Type srcType, Attribute expr) {
  if (dstType == srcType)
    return expr;
  return llvm::TypeSwitch<Type, Attribute>(dstType)
      .Case<IntegerType>(
          [&](IntegerType type) { return castToInt(type, srcType, expr); })
      .Case<FloatType>(
          [&](FloatType type) { return castToFloat(type, srcType, expr); })
      .Case<IndexType>(
          [&](IndexType type) { return castToIndex(type, srcType, expr); })
      .Default([](Type) { return nullptr; });
}

//===----------------------------------------------------------------------===//
// API definition
//===----------------------------------------------------------------------===//

Attribute xblang::eval::evalBinOp(BinaryOperator op, Attribute lhs,
                                  Attribute rhs, mlir::Builder &builder,
                                  Location loc) {
  TypedAttr lhsTA = dyn_cast_or_null<TypedAttr>(lhs);
  TypedAttr rhsTA = dyn_cast_or_null<TypedAttr>(rhs);
  // Return if the attributes don't have a type.
  if (!lhsTA || !rhsTA)
    return nullptr;
  // Return if the attributes don't have the same type.
  if (lhsTA.getType() != rhsTA.getType() ||
      !lhsTA.getType().isIntOrIndexOrFloat())
    return nullptr;
  return ArithEval(builder, lhs, rhs, lhsTA.getType(), loc).evalBinOp(op);
}

Attribute xblang::eval::evalCast(Type type, Attribute value,
                                 mlir::Builder &builder) {
  TypedAttr typedAttr = dyn_cast_or_null<TypedAttr>(value);
  // Return nullptr if the operands are not valid.
  if ((!typedAttr || !type) || (!type.isIntOrIndexOrFloat() ||
                                !typedAttr.getType().isIntOrIndexOrFloat()))
    return nullptr;
  // Return if there's no cast to perform.
  if (typedAttr.getType() == type)
    return value;
  return CastEval(builder).castValue(type, typedAttr.getType(), value);
}
