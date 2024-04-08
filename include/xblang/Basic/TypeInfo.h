//===- TypeInfo.h - Type Info ------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TypeInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_TYPEINFO_H
#define XBLANG_BASIC_TYPEINFO_H

#include <cstdint>
#include <limits>

namespace xblang {
class XBContext;
class TypeInfo;

struct TypeInfoBase {
  using ID = uint32_t;
  static constexpr ID uninitialized = std::numeric_limits<ID>::max();
};

/// Class for holding a static type ID.
template <typename T>
class StaticTypeInfo : public TypeInfoBase {
public:
  StaticTypeInfo() = default;

  bool operator==(const StaticTypeInfo &other) const {
    return other.getID() == getID();
  }

  /// Returns the type ID.
  static void *getID() { return &id; }

private:
  friend class XBContext;
  friend class TypeInfo;
  static ID id;
};

template <typename T>
TypeInfoBase::ID StaticTypeInfo<T>::id =
    std::numeric_limits<TypeInfoBase::ID>::max();

#define XB_DECLARE_TYPEINFO(x)                                                 \
  extern template class ::xblang::StaticTypeInfo<x>;
#define XB_DEFINE_TYPEINFO(x) template class ::xblang::StaticTypeInfo<x>;

/// Class for holding a type ID managed by a context.
class TypeInfo : public TypeInfoBase {
public:
  TypeInfo(const TypeInfo &) = default;
  TypeInfo(TypeInfo &&) = default;
  TypeInfo &operator=(const TypeInfo &) = default;
  TypeInfo &operator=(TypeInfo &&) = default;

  bool operator==(const TypeInfo &other) const { return other.id == id; }

  /// Returns the type info for the type `T`.
  template <typename T>
  static TypeInfo get() {
    return TypeInfo(StaticTypeInfo<T>::id);
  }

  /// Returns the type ID.
  ID getID() const { return id; }

  /// Returns true if the ID is uninitialized.
  bool isUninitialized() const { return id == uninitialized; }

private:
  TypeInfo(ID id) : id(id) {}
  friend class XBContext;
  ID id = uninitialized;
};
} // namespace xblang

#endif // XBLANG_BASIC_TYPEINFO_H
