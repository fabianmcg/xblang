//===- Builder.h - XLG builder  ----------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XLG builder.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_BUILDER_H
#define XBLANG_XLG_BUILDER_H

#include "mlir/IR/Builders.h"
#include "xblang/Basic/Context.h"
#include "xblang/Support/BuilderBase.h"
#include "xblang/Support/LLVM.h"
#include "xblang/XLG/IR/XLGDialect.h"

namespace xblang {
namespace xlg {
class XLGBuilder : public ::mlir::OpBuilder, public ::xblang::BuilderBase {
public:
  using OpBuilder = ::mlir::OpBuilder;
  using OpBuilder::OpBuilder;
  using ConceptType = ::xblang::xlg::ConceptType;
  using Op = ::mlir::Operation *;
  using Value = ::mlir::Value;
  using ValueList = ::mlir::SmallVector<::mlir::Value>;

  /// Create a builder with the given context.
  explicit XLGBuilder(XBContext *ctx, Listener *listener = nullptr)
      : OpBuilder(ctx->getMLIRContext(), listener), xbCtx(ctx) {}

  /// Create a builder and set the insertion point to the start of the region.
  explicit XLGBuilder(XBContext *ctx, mlir::Region *region,
                      Listener *listener = nullptr)
      : XLGBuilder(ctx, listener) {
    if (!region->empty())
      setInsertionPoint(&region->front(), region->front().begin());
  }

  explicit XLGBuilder(XBContext *ctx, mlir::Region &region,
                      Listener *listener = nullptr)
      : XLGBuilder(ctx, &region, listener) {}

  /// Create a builder and set insertion point to the given operation, which
  /// will cause subsequent insertions to go right before it.
  explicit XLGBuilder(XBContext *ctx, mlir::Operation *op,
                      Listener *listener = nullptr)
      : XLGBuilder(ctx, listener) {
    setInsertionPoint(op);
  }

  XLGBuilder(XBContext *ctx, mlir::Block *block,
             mlir::Block::iterator insertPoint, Listener *listener = nullptr)
      : XLGBuilder(ctx, listener) {
    setInsertionPoint(block, insertPoint);
  }

  /// Returns the XBLang context
  XBContext *getXBContext() const { return xbCtx; }

  /// Returns the concept stored in the context.
  template <typename Concept>
  Concept *getConcept() {
    return Concept::get(xbCtx);
  }

  /// Returns a class type with the specified concept.
  ConceptType getConceptClass(Concept *cncpt) {
    return ConceptType::get(getContext(), cncpt);
  }

  template <typename Concept>
  ConceptType getConceptClass() {
    return ConceptType::get(getContext(), getConcept<Concept>());
  }

  /// Returns a managed insertion block that sets the insertion point of the
  /// builder to the end of the new block.
  InsertionBlock getInsertionBlock() { return InsertionBlock(*this); }

  /// Returns an identifier.
  mlir::SymbolRefAttr getSymRef(mlir::StringRef sym) {
    return getAttr<mlir::SymbolRefAttr>(sym);
  }

  /// Creates an XLG template.
  TemplateOp createTemplate(mlir::Location loc, ConceptType kind,
                            mlir::Block *block,
                            ArrayRef<TemplateParam> parameters, Op ret);

protected:
  /// XBLang context.
  XBContext *xbCtx;
};

/// Class holding a reference to an XLGBuilder.
class XLGBuilderRef {
protected:
  XLGBuilder &builder;

public:
  XLGBuilderRef(XLGBuilder &builder) : builder(builder) {}

  /// Returns a class type with the specified concept.
  template <typename Concept>
  ConceptType getConceptClass() {
    return builder.getConceptClass<Concept>();
  }

  /// Returns a managed insertion block that sets the insertion point of the
  /// builder to the end of the new block.
  InsertionBlock getInsertionBlock() { return builder.getInsertionBlock(); }

  /// Creates a builder guard.
  template <typename... Args>
  XLGBuilder::Guard guard(Args &&...args) {
    return builder.guard(builder, std::forward<Args>(args)...);
  }
};
} // namespace xlg
} // namespace xblang

#endif // XBLANG_XLG_BUILDER_H
