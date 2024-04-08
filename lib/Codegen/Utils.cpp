//===- Utils.cpp - Common builder utilities ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common builder utility functions and classes.
//
//===----------------------------------------------------------------------===//

#include "xblang/Codegen/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace xblang;

namespace {
struct ArithGen {
  OpBuilder &builder;
  Value lhs;
  Value rhs;
  Type type;
  Location loc;

  ArithGen(OpBuilder &builder, Value lhs, Value rhs, Type type)
      : builder(builder), lhs(lhs), rhs(rhs), type(type), loc(lhs.getLoc()) {}

  /// Creates a binary operation.
  Value createBinOp(BinaryOperator op);
  /// Creates an add operation.
  Value createAddOp();
  /// Creates a sub operation.
  Value createSubOp();
  /// Creates a mul operation.
  Value createMulOp();
  /// Creates a div operation.
  Value createDivOp();
  /// Creates a mod operation.
  Value createModOp();
  /// Creates a left shift operation.
  Value createLShiftOp();
  /// Creates a right shift operation.
  Value createRShiftOp();
  /// Creates a compare equal operation.
  Value createCmpEqOp();
  /// Creates a compare not equal operation.
  Value createCmpNeqOp();
  /// Creates a compare less than operation.
  Value createCmpLtOp();
  /// Creates a compare greater than operation.
  Value createCmpGtOp();
  /// Creates a compare less equal operation.
  Value createCmpLeqOp();
  /// Creates a compare greater equal operation.
  Value createCmpGeqOp();
  /// Creates a compare spaceship operation.
  Value createCmpSpaceshipOp();
  /// Creates a binary and operation.
  Value createAndOp();
  /// Creates a binary or operation.
  Value createOrOp();
  /// Creates a binary xor operation.
  Value createXorOp();
};

struct CastGen {
  OpBuilder &builder;
  const TypeConverter *converter;
  Location loc;

  CastGen(OpBuilder &builder, const TypeConverter *converter, Location loc)
      : builder(builder), converter(converter), loc(loc) {}

  /// Converts a type.
  Type convert(Type type) const { return converter->convertType(type); }

  /// Casts a value to an integer type.
  Value castToInt(IntegerType type, Type srcType, Value expr);
  /// Casts a value to a float type.
  Value castToFloat(FloatType type, Type srcType, Value expr);
  /// Casts a value to an index type.
  Value castToIndex(IndexType type, Type srcType, Value expr);
  /// Casts a value.
  Value cast(Type dstType, Type srcType, Value expr);
};
} // namespace

//===----------------------------------------------------------------------===//
// ArithGen
//===----------------------------------------------------------------------===//

Value ArithGen::createAddOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
  else if (isa<FloatType>(type))
    return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::AddOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createMulOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
  else if (isa<FloatType>(type))
    return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::MulOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createSubOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
  else if (isa<FloatType>(type))
    return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::SubOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createDivOp() {
  if (auto iTy = dyn_cast<IntegerType>(type)) {
    if (iTy.isSigned())
      return builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::DivUIOp>(loc, lhs, rhs);
  } else if (isa<FloatType>(type))
    return builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::DivUOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createModOp() {
  if (auto iTy = dyn_cast<IntegerType>(type)) {
    if (iTy.isSigned())
      return builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::RemUIOp>(loc, lhs, rhs);
  } else if (isa<FloatType>(type))
    return builder.create<mlir::arith::RemFOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::RemUOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createLShiftOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::ShlOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createRShiftOp() {
  if (auto iTy = dyn_cast<IntegerType>(type)) {
    if (iTy.isSigned())
      return builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs);
  } else if (isa<IndexType>(type))
    return builder.create<mlir::index::ShrUOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createCmpEqOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  else if (isa<FloatType>(type))
    return builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::CmpOp>(
        loc, mlir::index::IndexCmpPredicate::EQ, lhs, rhs);
  return nullptr;
}

Value ArithGen::createCmpNeqOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
  else if (isa<FloatType>(type))
    return builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::CmpOp>(
        loc, mlir::index::IndexCmpPredicate::NE, lhs, rhs);
  return nullptr;
}

Value ArithGen::createCmpLtOp() {
  if (auto iTy = dyn_cast<IntegerType>(type)) {
    if (iTy.isSigned())
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, lhs, rhs);
  } else if (isa<FloatType>(type))
    return builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::CmpOp>(
        loc, mlir::index::IndexCmpPredicate::ULT, lhs, rhs);
  return nullptr;
}

Value ArithGen::createCmpGtOp() {
  if (auto iTy = dyn_cast<IntegerType>(type)) {
    if (iTy.isSigned())
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs);
  } else if (isa<FloatType>(type))
    return builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::CmpOp>(
        loc, mlir::index::IndexCmpPredicate::UGT, lhs, rhs);
  return nullptr;
}

Value ArithGen::createCmpLeqOp() {
  if (auto iTy = dyn_cast<IntegerType>(type)) {
    if (iTy.isSigned())
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ule, lhs, rhs);
  } else if (isa<FloatType>(type))
    return builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::CmpOp>(
        loc, mlir::index::IndexCmpPredicate::ULE, lhs, rhs);
  return nullptr;
}

Value ArithGen::createCmpGeqOp() {
  if (auto iTy = dyn_cast<IntegerType>(type)) {
    if (iTy.isSigned())
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::uge, lhs, rhs);
  } else if (isa<FloatType>(type))
    return builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::CmpOp>(
        loc, mlir::index::IndexCmpPredicate::UGE, lhs, rhs);
  return nullptr;
}

Value ArithGen::createCmpSpaceshipOp() {
  Value geq = ArithGen(builder, lhs, rhs, type).createCmpGeqOp();
  Value ceq = ArithGen(builder, lhs, rhs, type).createCmpEqOp();
  assert(geq && ceq && "invalid operands");
  auto lt = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIntegerType(8), -1));
  auto eq = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIntegerType(8), 0));
  auto gt = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIntegerType(8), 1));
  return builder.create<mlir::arith::SelectOp>(
      loc, geq, builder.create<mlir::arith::SelectOp>(loc, ceq, eq, gt), lt);
}

Value ArithGen::createAndOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::AndOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createOrOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::OrOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createXorOp() {
  if (isa<IntegerType>(type))
    return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
  else if (isa<IndexType>(type))
    return builder.create<mlir::index::XOrOp>(loc, lhs, rhs);
  return nullptr;
}

Value ArithGen::createBinOp(BinaryOperator op) {
  switch (op) {
  case BinaryOperator::Add:
    return createAddOp();
  case BinaryOperator::Sub:
    return createSubOp();
  case BinaryOperator::Mul:
    return createMulOp();
  case BinaryOperator::Div:
    return createDivOp();
  case BinaryOperator::Mod:
    return createModOp();
  case BinaryOperator::LShift:
    return createLShiftOp();
  case BinaryOperator::RShift:
    return createRShiftOp();
  case BinaryOperator::Equal:
    return createCmpEqOp();
  case BinaryOperator::NEQ:
    return createCmpNeqOp();
  case BinaryOperator::Less:
    return createCmpLtOp();
  case BinaryOperator::Greater:
    return createCmpGtOp();
  case BinaryOperator::LEQ:
    return createCmpLeqOp();
  case BinaryOperator::GEQ:
    return createCmpGeqOp();
  case BinaryOperator::Spaceship:
    return createCmpSpaceshipOp();
  case BinaryOperator::BinaryAnd:
    return createAndOp();
  case BinaryOperator::BinaryOr:
    return createOrOp();
  case BinaryOperator::BinaryXor:
    return createXorOp();
  default:
    break;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CastGen
//===----------------------------------------------------------------------===//

Value CastGen::castToInt(IntegerType tgtType, Type srcType, Value expr) {
  using namespace mlir;
  auto intType = convert(tgtType);
  if (tgtType.getWidth() == 1) {
    TypedAttr zero{};
    if (isa<IntegerType>(srcType))
      zero = builder.getIntegerAttr(convert(srcType), 0);
    else if (isa<FloatType>(srcType))
      zero = builder.getFloatAttr(convert(srcType), 0);
    else if (isa<IndexType>(srcType))
      zero = builder.getIndexAttr(0);
    else
      return nullptr;
    return ArithGen(builder, expr, builder.create<arith::ConstantOp>(loc, zero),
                    srcType)
        .createCmpNeqOp();
  }
  if (auto srcTy = dyn_cast<IntegerType>(srcType)) {
    if (tgtType.getWidth() < srcTy.getWidth())
      return builder.create<arith::TruncIOp>(loc, intType, expr);
    else if (tgtType.getWidth() == srcTy.getWidth())
      return expr;
    if (srcTy.isSigned())
      return builder.create<arith::ExtSIOp>(loc, intType, expr);
    return builder.create<arith::ExtUIOp>(loc, intType, expr);
  } else if (auto srcTy = dyn_cast<FloatType>(srcType)) {
    if (tgtType.isSigned())
      return builder.create<arith::FPToSIOp>(loc, intType, expr);
    return builder.create<arith::FPToUIOp>(loc, intType, expr);
  } else if (auto srcTy = dyn_cast<IndexType>(srcType)) {
    return builder.create<index::CastUOp>(loc, intType, expr);
  }
  return nullptr;
}

Value CastGen::castToFloat(FloatType tgtType, Type srcType, Value expr) {
  using namespace mlir;
  auto floatType = convert(tgtType);
  if (auto srcTy = dyn_cast<IntegerType>(srcType)) {
    if (srcTy.isSigned())
      return builder.create<arith::SIToFPOp>(loc, floatType, expr);
    return builder.create<arith::UIToFPOp>(loc, floatType, expr);
  } else if (auto srcTy = dyn_cast<FloatType>(srcType)) {
    if (tgtType.getWidth() < srcTy.getWidth())
      return builder.create<arith::TruncFOp>(loc, floatType, expr);
    return builder.create<arith::ExtFOp>(loc, floatType, expr);
  } else if (auto srcTy = dyn_cast<IndexType>(srcType)) {
    expr = builder.create<index::CastUOp>(loc, builder.getI64Type(), expr);
    return builder.create<arith::UIToFPOp>(loc, floatType, expr);
  }
  return nullptr;
}

Value CastGen::castToIndex(IndexType tgtType, Type srcType, Value expr) {
  using namespace mlir;
  if (auto srcTy = dyn_cast<IntegerType>(srcType)) {
    if (srcTy.isSigned())
      return builder.create<index::CastSOp>(loc, tgtType, expr);
    return builder.create<index::CastUOp>(loc, tgtType, expr);
  } else if (auto srcTy = dyn_cast<FloatType>(srcType)) {
    expr = builder.create<arith::FPToUIOp>(loc, builder.getI64Type(), expr);
    return builder.create<index::CastUOp>(loc, tgtType, expr);
  } else if (auto srcTy = dyn_cast<IndexType>(srcType)) {
    return expr;
  }
  return nullptr;
}

Value CastGen::cast(Type dstType, Type srcType, Value expr) {
  if (dstType == expr.getType())
    return expr;
  return llvm::TypeSwitch<Type, Value>(dstType)
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

Value xblang::createArithBinOp(OpBuilder &builder, BinaryOperator op, Value lhs,
                               Value rhs, Type type) {
  assert(lhs && rhs && "invalid operands");
  assert(lhs.getType() == rhs.getType() && "invalid operand types");
  if (!type)
    type = lhs.getType();
  return ArithGen(builder, lhs, rhs, type).createBinOp(op);
}

Value xblang::createCastOp(mlir::Type dstType, mlir::Type srcType,
                           mlir::Value expr, mlir::OpBuilder &builder,
                           CastInfo *converter) {
  if (srcType == dstType)
    return expr;
  assert(dstType && srcType && expr && "invalid operands");
  auto typeConverter = cast<codegen::CGCastInfo>(converter)->getTypeConverter();
  assert(typeConverter && "invalid type converter");
  return CastGen(builder, typeConverter, expr.getLoc())
      .cast(dstType, srcType, expr);
}
