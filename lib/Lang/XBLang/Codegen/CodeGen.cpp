//===- CodeGen.cpp - XBG code generator ---------------------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares function for registering code generation patterns.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Codegen/Codegen.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xblang/Basic/Pattern.h"
#include "xblang/Interfaces/Codegen.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Support/LLVM.h"

#include "xblang/Dialect/XBLang/IR/Type.h"

using namespace xblang;
using namespace xblang::xbg;

namespace {
class XGBCGInterface : public xblang::CodegenDialectInterface {
public:
  using xblang::CodegenDialectInterface::CodegenDialectInterface;

  mlir::LogicalResult
  populateCodegenPatterns(GenericPatternSet &patterns,
                          mlir::TypeConverter *converter) const override {
    if (!converter)
      return failure();
    populateTypeConversions(patterns.getMLIRContext(), converter);
    populateDeclCGPatterns(patterns, converter);
    populateExprCGPatterns(patterns, converter);
    populateStmtCGPatterns(patterns, converter);
    populateTypeCGPatterns(patterns, converter);
    return mlir::success();
  }

  void populateTypeConversions(MLIRContext *context,
                               mlir::TypeConverter *converter) const;
};
} // namespace

void XGBCGInterface::populateTypeConversions(
    MLIRContext *context, mlir::TypeConverter *converter) const {
  converter->addConversion([context](IntegerType type) -> mlir::Type {
    if (type.isSignless())
      return type;
    return IntegerType::get(context, type.getWidth());
  });
  converter->addConversion([converter](xb::ReferenceType type) -> mlir::Type {
    return xb::ReferenceType::get(converter->convertType(type.getPointee()),
                                  type.getMemorySpace());
  });
  converter->addConversion([converter](xb::PointerType type) -> mlir::Type {
    return xb::PointerType::get(converter->convertType(type.getPointee()),
                                type.getMemorySpace());
  });
  converter->addConversion([converter,
                            context](xb::RangeType type) -> mlir::Type {
    return xb::RangeType::get(context,
                              converter->convertType(type.getIteratorType()));
  });
  converter->addConversion(
      [context, converter](xb::StructType type) -> mlir::Type {
        SmallVector<Type, 5> members;
        for (auto ty : type.getMembers())
          members.push_back(converter->convertType(ty));
        return xb::StructType::get(context, members);
      });
  converter->addConversion([converter](xb::NamedType type) -> mlir::Type {
    if (type.getType() &&
        failed(type.setType(converter->convertType(type.getType()))))
      return nullptr;
    return type;
  });
}

void xblang::xbg::registerXBGCGInterface(mlir::DialectRegistry &registry) {
  registry.insert<XBGDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, XBGDialect *dialect) {
    dialect->addInterfaces<XGBCGInterface>();
  });
}

void xblang::xbg::registerXBGCGInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXBGCGInterface(registry);
  context.appendDialectRegistry(registry);
}
