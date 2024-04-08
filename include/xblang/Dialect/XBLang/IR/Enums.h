#ifndef XBLANG_DIALECT_XBLANG_IR_ENUMS_H
#define XBLANG_DIALECT_XBLANG_IR_ENUMS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

#include "xblang/Dialect/XBLang/IR/XBLangEnums.h.inc"

namespace xblang {
inline constexpr BinaryOperator operator~(BinaryOperator lhs) {
  return static_cast<BinaryOperator>(~static_cast<uint32_t>(lhs));
}

inline constexpr BinaryOperator operator|(BinaryOperator lhs,
                                          BinaryOperator rhs) {
  return static_cast<BinaryOperator>(static_cast<uint32_t>(lhs) |
                                     static_cast<uint32_t>(rhs));
}

inline constexpr BinaryOperator operator&(BinaryOperator lhs,
                                          BinaryOperator rhs) {
  return static_cast<BinaryOperator>(static_cast<uint32_t>(lhs) &
                                     static_cast<uint32_t>(rhs));
}

inline constexpr BinaryOperator operator^(BinaryOperator lhs,
                                          BinaryOperator rhs) {
  return static_cast<BinaryOperator>(static_cast<uint32_t>(lhs) ^
                                     static_cast<uint32_t>(rhs));
}

inline constexpr bool isValidOpKind(BinaryOperator kind) {
  bool isCompound =
      (kind & BinaryOperator::Compound) == BinaryOperator::Compound;
  BinaryOperator base = isCompound ? (kind ^ BinaryOperator::Compound) : kind;
  return (BinaryOperator::firstBinOp < base) &&
         (base < BinaryOperator::lastBinOp);
}

inline constexpr bool isCompoundOp(BinaryOperator kind) {
  return (kind & BinaryOperator::Compound) == BinaryOperator::Compound;
}

inline constexpr bool isAlgebraicOp(BinaryOperator kind) {
  switch (kind) {
  case BinaryOperator::Add:
  case BinaryOperator::Sub:
  case BinaryOperator::Mul:
  case BinaryOperator::Div:
  case BinaryOperator::Mod:
  case BinaryOperator::Pow:
    return true;
  default:
    return false;
  }
}

inline constexpr bool isComparisonOp(BinaryOperator kind) {
  switch (kind) {
  case BinaryOperator::Equal:
  case BinaryOperator::NEQ:
  case BinaryOperator::GEQ:
  case BinaryOperator::LEQ:
  case BinaryOperator::Less:
  case BinaryOperator::Greater:
  case BinaryOperator::Spaceship:
    return true;
  default:
    return false;
  }
}

inline constexpr bool isBitOp(BinaryOperator kind) {
  switch (kind) {
  case BinaryOperator::LShift:
  case BinaryOperator::RShift:
  case BinaryOperator::BinaryAnd:
  case BinaryOperator::BinaryOr:
  case BinaryOperator::BinaryXor:
    return true;
  default:
    return false;
  }
}

inline constexpr bool isLogicalOp(BinaryOperator kind) {
  switch (kind) {
  case BinaryOperator::And:
  case BinaryOperator::Or:
    return true;
  default:
    return false;
  }
}

inline constexpr BinaryOperator removeCompound(BinaryOperator kind) {
  return kind & ~BinaryOperator::Compound;
}

inline constexpr BinaryOperator addCompound(BinaryOperator kind) {
  return kind | BinaryOperator::Compound;
}
} // namespace xblang

#endif
