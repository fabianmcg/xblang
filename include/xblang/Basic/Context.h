//===- Context.h - XB compiler context ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XBLang compiler context.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_CONTEXT_H
#define XBLANG_BASIC_CONTEXT_H

#include "mlir/Support/TypeID.h"
#include "xblang/Basic/TypeInfo.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
class MLIRContext;
class Dialect;
} // namespace mlir

namespace xblang {
class XBContextImpl;
class Concept;
struct ModelInterface;

/// Class holding the XBLang context using the pImpl idiom.
class XBContext {
private:
  template <typename T>
  using void_t = void;

  template <typename, typename = void>
  struct declares_concept : std::false_type {};

  template <typename T>
  struct declares_concept<T, void_t<typename T::ConceptType>> : std::true_type {
  };

public:
  XBContext(mlir::MLIRContext &mlirContext);
  ~XBContext();
  XBContext(const XBContext &) = delete;
  XBContext &operator=(const XBContext &) = delete;

  /// Returns the MLIR context employed by the XBLang context.
  mlir::MLIRContext *getMLIRContext() const;

  /// Registers a concept with the context.
  template <typename ConceptTy, typename... Args>
  Concept *registerConcept(Args &&...args) {
    if (registerType<ConceptTy>() || !getConcept(TypeInfo::get<ConceptTy>()))
      return registerConceptImpl(
          new ConceptTy(this, std::forward<Args>(args)...));
    return nullptr;
  }

  /// Registers a parent concept with a concept in the context.
  template <typename ParentConcept>
  void registerParentConcept(TypeInfo c) {
    registerType<ParentConcept>();
    registerCast(c, TypeInfo::get<ParentConcept>());
  }

  /// Registers a construct in the context.
  template <typename Construct, typename... Constructs>
  void registerConstructs() {
    registerConstruct<Construct>();
    registerConstructs<Constructs...>();
  }

  template <typename... Constructs,
            std::enable_if_t<sizeof...(Constructs) == 0, int> = 0>
  void registerConstructs() {}

  template <typename Construct,
            std::enable_if_t<declares_concept<Construct>::value, int> = 0>
  void registerConstruct() {
    using ConceptType = typename Construct::ConceptType;
    using Model = typename ConceptType::template Model<Construct>;
    auto constructId = mlir::TypeID::get<Construct>();
    ConceptType *cep = ConceptType::get(this);
    assert(cep && "concept hasn't been registered");
    if (getInterface(constructId, TypeInfo::get<ConceptType>()))
      return;
    Model *iface = new Model();
    registerInterface(iface,
                      static_cast<typename ConceptType::PureInterface *>(iface),
                      constructId, cep);
    registerConstructParents(iface, constructId,
                             (typename ConceptType::ParentConcepts *)nullptr);
  }

  template <typename Construct,
            std::enable_if_t<!declares_concept<Construct>::value, int> = 0>
  void registerConstruct() {}

  /// Returns the registered concept under dialect and identifier.
  Concept *getConcept(llvm::StringRef dialect, llvm::StringRef identifier);
  Concept *getConcept(TypeInfo info);

  /// Returns whether the concept is of a certain class.
  template <typename ConceptTy>
  bool isConceptClass(Concept *const c) const {
    return isConceptClass(TypeInfo::get<ConceptTy>(), c);
  }

  /// Returns a raw construct interface.
  void *getInterface(mlir::TypeID constructId, TypeInfo info);

private:
  friend class Concept;
  /// Registers a concept with the context.
  Concept *registerConceptImpl(Concept *c);

  /// Registers a type with the context.
  template <typename T>
  bool registerType() {
    StaticTypeInfo<T> info;
    return registerType(info.id);
  }

  /// Returns whether a cast between types is possible.
  template <typename Dst, typename Src>
  bool isCastPossible() const {
    return isCastPossible(TypeInfo::get<Dst>(), TypeInfo::get<Src>());
  }

  bool isCastPossible(TypeInfo dst, TypeInfo src) const;

  /// Registers a cast.
  template <typename Dst, typename Src>
  bool registerCast() {
    return registerCast(TypeInfo::get<Dst>(), TypeInfo::get<Src>());
  }

  /// Registers a possible cast.
  void registerCast(TypeInfo dst, TypeInfo src);

  /// Returns whether a cast between types is possible.
  bool registerType(TypeInfo::ID &id);

  /// Registers a concept interface to a construct.
  void registerInterface(ModelInterface *iface, void *model,
                         mlir::TypeID constructId, Concept *cep);
  /// Attaches an interface to type ID type Info pair.
  void attachInterface(void *iface, mlir::TypeID constructId, TypeInfo info);

  /// Register the parent interfaces of a construct.
  template <typename Model, typename Concept, typename... Concepts>
  void registerConstructParents(Model *iface, mlir::TypeID constructId,
                                std::tuple<Concept, Concepts...> *) {
    attachInterface(static_cast<typename Concept::PureInterface *>(iface),
                    constructId, TypeInfo::get<Concept>());
    registerConstructParents(iface, constructId,
                             (typename Concept::ParentConcepts *)nullptr);
    registerConstructParents(iface, constructId,
                             (std::tuple<Concepts...> *)nullptr);
  }

  template <typename Model>
  void registerConstructParents(Model *iface, mlir::TypeID constructId,
                                std::tuple<> *) {}
  /// Pointer to the context implementation.
  const std::unique_ptr<XBContextImpl> impl;
};

/// Returns an XB context stored in the dialect.
XBContext *getXBContext(mlir::Dialect *dialect);

/// Returns an XB context stored in the MLIR context.
XBContext *getXBContext(mlir::MLIRContext *context);
} // namespace xblang

#endif // XBLANG_BASIC_CONTEXT_H
