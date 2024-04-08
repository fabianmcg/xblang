//===- Codegen.h - Code generation -------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares code generation functions and classes.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_CODEGEN_CODEGEN_H
#define XBLANG_CODEGEN_CODEGEN_H

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "xblang/Basic/Pattern.h"
#include "xblang/Basic/TypeSystem.h"
#include "xblang/Support/BuilderBase.h"
#include "xblang/Support/LLVM.h"

namespace mlir {
class TypeConverter;
}

namespace xblang {
class XBContext;

namespace codegen {
class CGDriver;
/// Applies the code generation driver to op.
LogicalResult
applyCodegenDriver(mlir::Operation *op, const FrozenPatternSet &patterns,
                   const mlir::TypeConverter *typeConverter = nullptr);

/// Result returned by CG pattern.
using CGResult = llvm::PointerUnion<Operation *, Value, Attribute, Type>;

namespace detail {
class CGDriverImpl;
}

/// Class for passing the type converter to the type system.
class CGCastInfo : public CastInfoMixin<CGCastInfo> {
public:
  /// Converts a type.
  Type convertType(Type type) const;

  /// Returns the type converter.
  const mlir::TypeConverter *getTypeConverter() const { return typeConverter; }

private:
  friend class CGDriver;

  CGCastInfo(const mlir::TypeConverter *typeConverter)
      : typeConverter(typeConverter) {}

  const mlir::TypeConverter *typeConverter;
};

/// Code generation driver.
class CGDriver : public PatternRewriter, public BuilderBase {
public:
  using PatternRewriter::PatternRewriter;
  CGDriver(XBContext *context, const FrozenPatternSet &patterns,
           const mlir::TypeConverter *typeConverter);

  /// Generates a new op.
  CGResult genOp(Operation *op);

  CGResult genValue(Value value) {
    auto [result, found] = lookup(value);
    if (found)
      return result;
    if (auto op = value.getDefiningOp())
      return map(value, genOp(op));
    return nullptr;
  }

  /// This method erases a block.
  void eraseBlock(Block *block) override;

  /// This method erases an operation.
  void eraseOp(Operation *op) override;

  /// This method replaces an operation.
  void replaceOp(Operation *op, Operation *newOp) override;
  void replaceOp(Operation *op, ValueRange newValues) override;

  /// Returns the XB context.
  XBContext *getXBContext() const { return context; }

  /// Returns a concept interface for the given operation.
  template <typename Interface>
  Interface getInterface(mlir::Operation *op) {
    return Interface::get(getXBContext(), op);
  }

  /// Returns the driver implementation.
  detail::CGDriverImpl *getImpl() { return impl.get(); }

  /// Returns the current CG value for a given key.
  std::pair<CGResult, bool> lookup(CGResult key) {
    auto it = cgMapping.find(key);
    return it != cgMapping.end() ? std::pair<CGResult, bool>{it->second, true}
                                 : std::pair<CGResult, bool>{nullptr, false};
  }

  template <typename T>
  T lookup(CGResult key) {
    auto it = cgMapping.find(key);
    return it != cgMapping.end() ? it->second.dyn_cast<T>() : nullptr;
  }

  /// Returns the current value for a given key.
  Value lookupValue(CGResult key) {
    auto it = valueMapping.find(key);
    return it != valueMapping.end() ? it->second : nullptr;
  }

  /// Maps a key to a value in the current scope.
  Value mapValue(CGResult key, Value value) {
    return valueMapping[key] = value;
  }

  /// Allow access to the type system.
  TypeSystem *operator->() { return &typeSystem; }

  const TypeSystem *operator->() const { return &typeSystem; }

  /// Returns the cast info used by the type system in CG.
  CGCastInfo *getCastInfo() { return &castInfo; }

  /// Makes a cast.
  Value makeCast(Type tgt, Type src, Value value) {
    return typeSystem.makeCast(tgt, src, value, *this, &castInfo);
  }

private:
  friend class detail::CGDriverImpl;
  std::unique_ptr<detail::CGDriverImpl> impl{};
  XBContext *context;

  /// Maps a key to a CG value in the current scope.
  CGResult map(CGResult key, CGResult value) { return cgMapping[key] = value; }

protected:
  const mlir::TypeConverter *typeConverter{};
  TypeSystem typeSystem{};
  DenseMap<CGResult, CGResult> cgMapping{};
  DenseMap<CGResult, Value> valueMapping{};
  CGCastInfo castInfo;
};

//===----------------------------------------------------------------------===//
// Codegen Patterns
//===----------------------------------------------------------------------===//

/// Base class for the code generation patterns.
class CGPattern : public GenericPattern {
protected:
  using GenericPattern::GenericPattern;

  template <typename TagOrStringRef>
  CGPattern(TagOrStringRef arg, const TypeConverter *typeConverter,
            PatternBenefit benefit, MLIRContext *context,
            ArrayRef<StringRef> generatedNames = {})
      : GenericPattern(arg, benefit, context, generatedNames),
        typeConverter(typeConverter) {}

  template <typename Tag>
  CGPattern(Tag tag, TypeID typeID, const TypeConverter *typeConverter,
            PatternBenefit benefit, MLIRContext *context,
            ArrayRef<StringRef> generatedNames = {})
      : GenericPattern(tag, typeID, benefit, context, generatedNames),
        typeConverter(typeConverter) {}

public:
  /// Generates an op, returning a.
  virtual CGResult generate(mlir::Operation *op, CGDriver &driver) const = 0;

  /// Converts a type if possible, asserts if the type converter is missing.
  Type convertType(Type type) const;

protected:
  /// Type converter.
  const TypeConverter *typeConverter{};
};

/// Class for code generation patterns based on a concrete op.
template <typename SourceOp>
class OpCGPattern : public CGPattern {
public:
  using Base = OpCGPattern;
  using Op = SourceOp;
  using OpAdaptor = typename Op::Adaptor;

  OpCGPattern(MLIRContext *context,
              const TypeConverter *typeConverter = nullptr,
              PatternBenefit benefit = 1)
      : CGPattern(Op::getOperationName(), typeConverter, benefit, context) {}

  /// Wrappers around the CGPattern methods that pass the derived op
  /// type.
  LogicalResult match(Operation *op) const final { return match(cast<Op>(op)); }

  CGResult generate(mlir::Operation *op, CGDriver &driver) const final {
    return generate(cast<Op>(op), driver);
  }

  /// Concrete op specific methods.
  virtual LogicalResult match(Op op) const { return success(); }

  virtual CGResult generate(Op op, CGDriver &driver) const { return nullptr; }
};

/// Class for code generation patterns based on an interface.
template <typename Iface>
class InterfaceCGPattern : public CGPattern {
public:
  using Base = InterfaceCGPattern;
  using Interface = Iface;

  InterfaceCGPattern(MLIRContext *context,
                     const TypeConverter *typeConverter = nullptr,
                     PatternBenefit benefit = 1)
      : CGPattern(Pattern::MatchInterfaceOpTypeTag(), TypeID::get<Interface>(),
                  typeConverter, benefit, context) {}

  /// Wrappers around the CGPattern methods that pass the derived interface
  /// type.
  LogicalResult match(Operation *op) const final {
    return match(cast<Interface>(op));
  }

  CGResult generate(mlir::Operation *op, CGDriver &driver) const final {
    return generate(cast<Interface>(op), driver);
  }

  /// Concrete interface specific methods.
  virtual LogicalResult match(Interface op) const { return success(); }

  virtual CGResult generate(Interface op, CGDriver &driver) const {
    return nullptr;
  }
};
} // namespace codegen
} // namespace xblang

MLIR_DECLARE_EXPLICIT_TYPE_ID(xblang::codegen::CGCastInfo);

#endif // XBLANG_CODEGEN_CODEGEN_H
