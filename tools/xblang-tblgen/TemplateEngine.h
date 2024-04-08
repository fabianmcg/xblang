//===- TemplateEngine.h - Text template engine -------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the template text engine.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_TBLGEN_TEMPLATEENGINE_H
#define XBLANG_TBLGEN_TEMPLATEENGINE_H

#include "xblang/Support/Format.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace xblang {
namespace tablegen {
/// Base class for all all text templates.
class TextTemplate {
public:
  using Environment = llvm::StringMap<std::shared_ptr<TextTemplate>>;
  using ID = const void *;
  virtual ~TextTemplate() = default;
  /// Compile the template to a string.
  virtual std::string compile(const Environment &environment = {}) const = 0;

  /// Returns the class id.
  ID getID() const { return id; }

  /// Returns whether the template is of the appropriate class.
  static inline bool classof(TextTemplate const *tmpl) { return true; }

  /// Helper function for getting a value or null from an environment.
  static TextTemplate *get(llvm::StringRef key,
                           const Environment &environment) {
    auto it = environment.find(key);
    if (it != environment.end())
      return it->second.get();
    return nullptr;
  }

  /// Inserts a value to an environment.
  template <typename T,
            std::enable_if_t<std::is_base_of_v<TextTemplate, T>, int> = 0>
  static T *insert(Environment &environment, llvm::StringRef key, T &&value) {
    auto &ptr = environment[key] =
        std::unique_ptr<TextTemplate>(new T(std::move(value)));
    return static_cast<T *>(ptr.get());
  }

protected:
  TextTemplate(ID id) : id(id) {}

private:
  ID id;
};

/// Value text templates.
template <typename T>
class ValueTemplate : public TextTemplate {
public:
  ValueTemplate(ValueTemplate &&) = default;
  ValueTemplate(const ValueTemplate &) = default;
  ValueTemplate &operator=(ValueTemplate &&) = default;
  ValueTemplate &operator=(const ValueTemplate &) = default;

  static ValueTemplate make(T &&value, llvm::StringRef fmtString = "{0}") {
    return ValueTemplate(fmtString, std::move(value));
  }

  /// Uses llvm::format to return
  std::string compile(const Environment &environment = {}) const override {
    return fmt(fmtString.data(), value);
  }

  /// Returns whether the template is of the appropriate class.
  static inline bool classof(TextTemplate const *tmpl) {
    return tmpl->getID() == &const_cast<int &>(id);
  }

private:
  ValueTemplate(llvm::StringRef str, T &&value)
      : TextTemplate(&id), fmtString(str), value(std::move(value)) {}

  /// String used to format.
  llvm::StringRef fmtString;
  /// Value to format.
  T value;
  /// Stub class id.
  static int id;
};

template <typename V>
int ValueTemplate<V>::id = 0;

/// String template.
class StrTemplate : public TextTemplate {
public:
  StrTemplate(StrTemplate &&) = default;
  StrTemplate(const StrTemplate &) = default;
  StrTemplate &operator=(StrTemplate &&) = default;
  StrTemplate &operator=(const StrTemplate &) = default;

  static StrTemplate make(llvm::StringRef str) { return StrTemplate(str); }

  /// Uses llvm::format to return
  std::string compile(const Environment &environment = {}) const override {
    return str;
  }

  /// Returns whether the template is of the appropriate class.
  static inline bool classof(TextTemplate const *tmpl) {
    return tmpl->getID() == &const_cast<int &>(id);
  }

private:
  StrTemplate(llvm::StringRef str) : TextTemplate(&id), str(str.str()) {}

  /// String value.
  std::string str;
  /// Stub class id.
  static int id;
};

/// Template engine.
class TemplateEngine : public TextTemplate {
public:
  TemplateEngine(TemplateEngine &&) = default;
  TemplateEngine(const TemplateEngine &) = delete;
  TemplateEngine &operator=(TemplateEngine &&) = default;
  TemplateEngine &operator=(const TemplateEngine &) = delete;

  static TemplateEngine make(llvm::StringRef str) {
    return TemplateEngine(str.str());
  }

  TextTemplate *operator[](llvm::StringRef key) const {
    return get(key, environment);
  }

  /// Compile the template to a string.
  std::string compile(const Environment &environment = {}) const override;

  /// Returns whether the template is of the appropriate class.
  static inline bool classof(TextTemplate const *tmpl) {
    return tmpl->getID() == &const_cast<int &>(id);
  }

  /// Erases a value from the environment.
  void erase(llvm::StringRef key) { environment.erase(key); }

  /// Inserts a value to the environment.
  template <typename T,
            std::enable_if_t<std::is_base_of_v<TextTemplate, T>, int> = 0>
  T *insert(llvm::StringRef key, T &&value) {
    auto &ptr = environment[key] =
        std::unique_ptr<TextTemplate>(new T(std::move(value)));
    return static_cast<T *>(ptr.get());
  }

  /// Returns the environment used by the engine.
  const Environment &getEnvironment() const { return environment; }

private:
  TemplateEngine(std::string &&str) : TextTemplate(&id), tmpl(std::move(str)) {}

  /// Template string.
  std::string tmpl;
  /// Engine environment.
  Environment environment;
  /// Stub class id.
  static int id;
};
} // namespace tablegen
} // namespace xblang

#endif // XBLANG_TBLGEN_TEMPLATEENGINE_H
