//===- Type.cpp - Type system declaration ------------------------*- C++-*-===//
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

#include "xblang/Basic/TypeSystem.h"
#include "xblang/Sema/TypeUtil.h"

using namespace xblang;

//===----------------------------------------------------------------------===//
// TypeSystem
//===----------------------------------------------------------------------===//

bool TypeSystem::addCast(GenericType target, GenericType source,
                         CastFunction &&cast) {
  if (cast)
    return allowedCasts.insert({{target, source}, std::move(cast)}).second;
  return false;
}

bool TypeSystem::isValidCast(Type target, Type source) const {
  return target && source &&
         (target == source || isValidPrimitiveCast(target, source) ||
          allowedCasts.contains({target, source}));
}

bool TypeSystem::isValidCast(TypeClass target, TypeClass source) const {
  auto tk = target.getKey();
  auto sk = source.getKey();
  return tk && sk && allowedCasts.contains({tk, sk});
}

Value TypeSystem::makeCast(Type target, Type source, Value sourceValue,
                           OpBuilder &builder, CastInfo *info) const {
  if (target == source)
    return sourceValue;
  if (isValidPrimitiveCast(target, source) && primitiveCast)
    return makePrimitiveCast(target, source, sourceValue, builder, info);
  auto it = allowedCasts.find({target, source});
  if (it != allowedCasts.end())
    return it->second(target, source, sourceValue, builder, info);
  return nullptr;
}

Value TypeSystem::makeCast(TypeClass target, TypeClass source,
                           Value sourceValue, OpBuilder &builder,
                           CastInfo *info) const {
  auto it = allowedCasts.find({target.getKey(), source.getKey()});
  if (it != allowedCasts.end())
    return it->second(target.getType(), source.getType(), sourceValue, builder,
                      info);
  return nullptr;
}
