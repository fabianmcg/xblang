//===- Common.h - Common Tablegen classes ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common tablegen classes.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_TABLEGEN_COMMON_H
#define XBLANG_TABLEGEN_COMMON_H

#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"
#include <optional>

namespace xblang {
namespace tablegen {
using Trait = ::mlir::tblgen::Trait;
using NativeTrait = ::mlir::tblgen::NativeTrait;

/// Class for holding DAG args.
class Parameter {
public:
  /// Returns the name of this parameter.
  std::optional<llvm::StringRef> getName() const { return name; }

  /// Returns the def init of this argument.
  const llvm::Init *getInit() const { return def; }

  Parameter(std::optional<llvm::StringRef> name, const llvm::Init *def)
      : name(name), def(def) {}

protected:
  /// Name of the parameter.
  std::optional<llvm::StringRef> name;
  /// The tablegen definition of the parameter.
  const llvm::Init *def;
};

/// Class for holding Cpp args.
class CppParameter : public Parameter {
public:
  /// Return a string containing the C++ type of this parameter.
  llvm::StringRef getCppType() const;

  /// Return an optional string containing the default value to use for this
  /// parameter.
  std::optional<llvm::StringRef> getDefaultValue() const;

  /// Creates a CppParameter.
  static std::optional<CppParameter> get(Parameter param) {
    if (isValid(param.getInit()))
      return CppParameter(param.getName(), param.getInit());
    return std::nullopt;
  }

  /// Returns whether the CppParam is valid.
  static bool isValid(const llvm::Init *def);

  bool isValid() const { return isValid(def); }

private:
  using Parameter::Parameter;
};

/// Class for holding DAGs.
class ParameterList {
public:
  /// Returns an argument from the list.
  Parameter getParameter(size_t i) const {
    assert(i <= size() && "invalid index");
    if (llvm::StringInit *init = def->getArgName(i))
      return Parameter(init->getValue(), def->getArg(i));
    return Parameter(std::nullopt, def->getArg(i));
  }

  std::optional<CppParameter> getCppParameter(size_t i) const {
    return CppParameter::get(getParameter(i));
  }

  /// Returns the number of arguments in the list.
  size_t size() const { return def ? def->getNumArgs() : 0; }

  /// Returns a parameter list.
  static ParameterList get(llvm::DagInit *init) { return ParameterList(init); }

  /// Returns a vector with the parameters in tge list.
  llvm::SmallVector<Parameter> getList() const {
    llvm::SmallVector<Parameter> list;
    for (size_t i = 0; i < size(); ++i)
      list.push_back(getParameter(i));
    return list;
  }

  llvm::SmallVector<std::optional<CppParameter>> getCppList() const {
    llvm::SmallVector<std::optional<CppParameter>> list;
    for (size_t i = 0; i < size(); ++i)
      list.push_back(getCppParameter(i));
    return list;
  }

  /// Returns the DagInit definition.
  const llvm::DagInit *getDef() const { return def; }

private:
  ParameterList(const llvm::DagInit *def) : def(def) {}

  /// The tablegen definition of the parameter.
  const llvm::DagInit *def;
};
} // namespace tablegen
} // namespace xblang

#endif // XBLANG_TABLEGEN_COMMON_H
