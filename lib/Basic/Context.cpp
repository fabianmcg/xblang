//===- Context.cpp - XB compiler context -------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the the XBLang compiler context.
//
//===----------------------------------------------------------------------===//

#include "xblang/Basic/Context.h"
#include "mlir/IR/MLIRContext.h"
#include "xblang/Basic/Concept.h"
#include "xblang/Basic/ContextDialect.h"
#include "xblang/Basic/SourceManager.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

using namespace xblang;

//===----------------------------------------------------------------------===//
// XBContextImpl implementation
//===----------------------------------------------------------------------===//
class xblang::XBContextImpl {
public:
  friend class XBContext;
  using CastInfo = std::pair<TypeInfo::ID, TypeInfo::ID>;
  XBContextImpl(mlir::MLIRContext &mlirContext);

  // Returns an ordered CastInfo pair.
  static CastInfo getInfo(TypeInfo::ID dst, TypeInfo::ID src);

private:
  // MLIR context.
  mlir::MLIRContext *mlirContext;
  // Cast information.
  llvm::DenseSet<CastInfo> castInfo;
  // The next id to use when registering a type.
  TypeInfo::ID nextValidId = 0;
  // Concept map.
  llvm::StringMap<std::unique_ptr<Concept>> conceptMap;
  // Concept map by ID.
  llvm::DenseMap<TypeInfo::ID, Concept *> conceptIdMap;
  // Registry of all interfaces.
  llvm::DenseMap<std::pair<mlir::TypeID, TypeInfo::ID>,
                 std::unique_ptr<ModelInterface>>
      interfaceRegistry;
  // Map for retriving interfaces.
  llvm::DenseMap<std::pair<mlir::TypeID, TypeInfo::ID>, void *> interfaceMap;
};

XBContextImpl::XBContextImpl(mlir::MLIRContext &mlirContext)
    : mlirContext(&mlirContext) {}

XBContextImpl::CastInfo XBContextImpl::getInfo(TypeInfo::ID dst,
                                               TypeInfo::ID src) {
  if (src <= dst)
    return CastInfo(src, dst);
  return CastInfo(dst, src);
}

//===----------------------------------------------------------------------===//
// XBContext
//===----------------------------------------------------------------------===//
XBContext::XBContext(mlir::MLIRContext &mlirContext)
    : impl(new XBContextImpl(mlirContext)) {}

XBContext::~XBContext() = default;

mlir::MLIRContext *XBContext::getMLIRContext() const {
  return impl->mlirContext;
}

bool XBContext::isCastPossible(TypeInfo dst, TypeInfo src) const {
  assert(!dst.isUninitialized() && "the type hasn't been registered");
  assert(!src.isUninitialized() && "the type hasn't been registered");
  return impl->castInfo.contains(
      XBContextImpl::getInfo(dst.getID(), src.getID()));
}

void XBContext::registerCast(TypeInfo dst, TypeInfo src) {
  assert(!dst.isUninitialized() && "the type hasn't been registered");
  assert(!src.isUninitialized() && "the type hasn't been registered");
  impl->castInfo.insert(XBContextImpl::getInfo(dst.getID(), src.getID()));
}

void *XBContext::getInterface(mlir::TypeID constructId, TypeInfo info) {
  auto it = impl->interfaceMap.find({constructId, info.getID()});
  if (it != impl->interfaceMap.end())
    return it->second;
  return nullptr;
}

void XBContext::registerInterface(ModelInterface *iface, void *model,
                                  mlir::TypeID constructId, Concept *cep) {
  impl->interfaceRegistry[{constructId, cep->getTypeInfo().getID()}] =
      std::unique_ptr<ModelInterface>(iface);
  attachInterface(model, constructId, cep->getTypeInfo());
}

void XBContext::attachInterface(void *iface, mlir::TypeID constructId,
                                TypeInfo info) {
  if (iface)
    impl->interfaceMap[{constructId, info.getID()}] = iface;
}

bool XBContext::registerType(TypeInfo::ID &id) {
  if (id == TypeInfo::uninitialized) {
    id = impl->nextValidId++;
    return true;
  }
  return false;
}

Concept *XBContext::getConcept(llvm::StringRef dialect,
                               llvm::StringRef identifier) {
  auto it = impl->conceptMap.find((dialect + "::" + identifier).str());
  if (it != impl->conceptMap.end())
    return it->second.get();
  return nullptr;
}

Concept *XBContext::getConcept(TypeInfo info) {
  auto it = impl->conceptIdMap.find(info.getID());
  if (it != impl->conceptIdMap.end())
    return it->second;
  return nullptr;
}

Concept *XBContext::registerConceptImpl(Concept *c) {
  impl->conceptMap[(c->getDialect() + "::" + c->getIdentifier()).str()] =
      std::unique_ptr<Concept>(c);
  impl->conceptIdMap[c->getTypeInfo().getID()] = c;
  return c;
}

//===----------------------------------------------------------------------===//
// XBContextDialect
//===----------------------------------------------------------------------===//

XBContextDialect::XBContextDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
                      ::mlir::TypeID::get<XBContextDialect>()),
      ctx(*context) {}

XBContextDialect::~XBContextDialect() = default;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::xblang::XBContextDialect)

//===----------------------------------------------------------------------===//
// XBContextDialect
//===----------------------------------------------------------------------===//

XBContext *XBContextDialectInterface::geXBContext() {
  assert(getContext() && "invalid MLIR context");
  return xblang::getXBContext(getContext());
}

XBContext *xblang::getXBContext(mlir::Dialect *dialect) {
  if (auto ifce = llvm::dyn_cast<XBContextDialectInterface>(dialect))
    return ifce->geXBContext();
  return nullptr;
}

XBContext *xblang::getXBContext(mlir::MLIRContext *context) {
  if (!context)
    return nullptr;
  if (auto dialect = context->getOrLoadDialect<XBContextDialect>())
    return &(dialect->getContext());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Concept
//===----------------------------------------------------------------------===//
Concept::~Concept() = default;

bool Concept::isClassof(TypeInfo info, Concept const *cep) {
  if (cep) {
    if (cep->getTypeInfo() == info)
      return true;
    if (auto ctx = cep->getContext())
      return ctx->isCastPossible(info, cep->getTypeInfo());
  }
  return false;
}

mlir::MLIRContext *Concept::getMLIRContext() const {
  return context ? context->getMLIRContext() : nullptr;
}

Concept *Concept::get(XBContext *context, TypeInfo typeInfo) {
  if (context)
    return context->getConcept(typeInfo);
  return nullptr;
}

void *Concept::getRawInterface(mlir::TypeID constructID, TypeInfo info) {
  return context ? context->getInterface(constructID, info) : nullptr;
}
