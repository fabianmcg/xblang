//===- Type.h - Type system declaration --------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the type system driver.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_TYPESYSTEM_H
#define XBLANG_BASIC_TYPESYSTEM_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xblang/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

namespace xblang {
/// Class for passing additional information to a cast.
class CastInfo {
public:
  /// Returns the cast info kind.
  mlir::TypeID getKind() const { return kind; }

protected:
  CastInfo(mlir::TypeID kind) : kind(kind) {}

private:
  mlir::TypeID kind;
};

/// Helper class for easily implementing CastInfo derived types.
template <typename Derived>
class CastInfoMixin : public CastInfo {
public:
  CastInfoMixin() : CastInfo(mlir::TypeID::get<Derived>()) {}

  static inline bool classof(CastInfo const *info) {
    return info->getKind() == mlir::TypeID::get<Derived>();
  }
};

/// Class for managing a type system of casts and promotions. Casts are
/// permitted between instances of specific types or between classes of types.
class TypeSystem {
public:
  using GenericType = llvm::PointerUnion<Type, mlir::TypeID>;

  /// Represents a type class.
  class TypeClass {
  private:
    llvm::PointerIntPair<Type, 1, bool> type;

  public:
    TypeClass(Type type, bool isClass = false)
        : type(type, type ? isClass : false) {}

    /// Returns the type.
    Type getType() const { return type.getPointer(); }

    /// Returns the key used to identify this type.
    GenericType getKey() const {
      if (Type typePtr = type.getPointer()) {
        if (type.getInt())
          return typePtr.getTypeID();
        return typePtr;
      }
      return nullptr;
    }
  };

protected:
  template <typename FnTy, int I>
  using GetArg =
      typename llvm::function_traits<std::decay_t<FnTy>>::template arg_t<I>;
  using CastFunction =
      std::function<Value(Type, Type, Value, OpBuilder &, CastInfo *)>;
  using CastFunctionRef =
      llvm::function_ref<Value(Type, Type, Value, OpBuilder &, CastInfo *)>;
  using IsCastFunctionRef = llvm::function_ref<bool(Type, Type)>;
  using TargetSource = std::pair<GenericType, GenericType>;

  /// Map of allowed casts and their implementation.
  DenseMap<TargetSource, CastFunction> allowedCasts;

  /// A function pointer to a static function performing primitive casts.
  IsCastFunctionRef isPrimitiveCast{};

  /// A function pointer to a static function performing primitive casts.
  CastFunctionRef primitiveCast{};

public:
  TypeSystem() = default;

  /// Tries to set the function used for performing primitive casts.
  bool setPrimitiveCast(IsCastFunctionRef isCastFn, CastFunctionRef castFn) {
    bool set = false;
    if (!isPrimitiveCast) {
      isPrimitiveCast = isCastFn;
      set = true;
    }
    if (!primitiveCast) {
      primitiveCast = castFn;
      set = true;
    }
    return set;
  }

  /// Inserts a cast to the type system, returns true if the cast was inserted.
  bool addCast(GenericType target, GenericType source, CastFunction &&cast);

  template <typename FnTy, typename TargetTy = GetArg<FnTy, 0>,
            typename SourceTy = GetArg<FnTy, 1>>
  bool addCast(GenericType target, GenericType source, FnTy &&callback) {
    return addCast(
        target, source,
        wrapCastFunction<TargetTy, SourceTy>(std::forward<FnTy>(callback)));
  }

  /// Returns true if a cast is valid primitive cast, ie. cast between
  /// fundamental types in the type system.
  bool isValidPrimitiveCast(Type target, Type source) const {
    assert(isPrimitiveCast && "function was never set");
    return isPrimitiveCast(target, source);
  }

  /// Casts a value into a target type, where both the source and destination
  /// are primitive values, returns nullptr if the cast is invalid.
  Value makePrimitiveCast(Type target, Type source, Value sourceValue,
                          OpBuilder &builder, CastInfo *info = nullptr) const {
    assert(primitiveCast && "function was never set");
    return primitiveCast(target, source, sourceValue, builder, info);
  }

  /// Returns true if a cast is valid.
  bool isValidCast(Type target, Type source) const;
  bool isValidCast(TypeClass target, TypeClass source) const;
  /// Casts a value into a target type, returns nullptr if the cast is invalid.
  Value makeCast(Type target, Type source, Value sourceValue,
                 OpBuilder &builder, CastInfo *info = nullptr) const;
  Value makeCast(TypeClass target, TypeClass source, Value sourceValue,
                 OpBuilder &builder, CastInfo *info = nullptr) const;

  Value makeCast(Type target, Value source, OpBuilder &builder,
                 CastInfo *info = nullptr) const {
    return makeCast(target, source.getType(), source, builder, info);
  }

  Value makeCast(TypeClass target, Value source, OpBuilder &builder,
                 CastInfo *info = nullptr) const {
    return makeCast(target, source.getType(), source, builder, info);
  }

private:
  /// Wraps a cast function.
  template <typename TargetTy, typename SourceTy, typename FnTy>
  std::enable_if_t<
      std::is_invocable_v<FnTy, TargetTy, SourceTy, Value, OpBuilder &>,
      CastFunction>
  wrapCastFunction(FnTy &&callback) const {
    return [callback = std::forward<FnTy>(callback)](
               Type target, Type source, Value sourceValue, OpBuilder &builder,
               CastInfo *info) -> Value {
      TargetTy tgt = dyn_cast<TargetTy>(target);
      SourceTy src = dyn_cast<SourceTy>(source);
      if (!tgt || !src)
        return nullptr;
      return callback(tgt, src, sourceValue, builder, info);
    };
  }
};
} // namespace xblang

#endif // XBLANG_BASIC_TYPESYSTEM_H
