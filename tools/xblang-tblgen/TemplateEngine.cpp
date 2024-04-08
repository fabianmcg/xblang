//===- TemplateEngine.cpp - Text template engine -----------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the template text engine.
//
//===----------------------------------------------------------------------===//

#include "TemplateEngine.h"
#include "llvm/ADT/SmallString.h"

using namespace xblang::tablegen;

int StrTemplate::id = 0;
int TemplateEngine::id = 0;

std::string TemplateEngine::compile(const Environment &environment) const {
  if (this->tmpl.empty())
    return {};
  llvm::SmallString<128> result;
  llvm::StringMap<std::string> cache;
  llvm::StringRef tmpl = this->tmpl;
  const char *ptr = tmpl.data();
  const char *end = tmpl.data() + tmpl.size();
  // Helper function that tries to replace an identifier, returns true if it
  // succeeds.
  auto tryReplace = [&](llvm::StringRef identifier, const char *end) -> bool {
    // Check the cache.
    auto it = cache.find(identifier);
    if (it != cache.end()) {
      result.append(it->second);
      ptr = end;
      return true;
    }
    // Check the root environment.
    if (TextTemplate *txt = get(identifier, environment)) {
      result.append(cache[identifier] = txt->compile(environment));
      ptr = end;
      return true;
    }
    // Check the template's environment.
    if (TextTemplate *txt = get(identifier, this->environment)) {
      result.append(cache[identifier] = txt->compile(environment));
      ptr = end;
      return true;
    }
    return false;
  };
  auto validIdChar = [](char c) { return isalnum(c) || c == '_'; };
  auto get = [&end](const char *ptr) -> char {
    return ptr < end ? *ptr : static_cast<char>(0);
  };
  // Main routine.
  while (ptr < end) {
    if (*ptr == '$') {
      const char *idEnd = ptr + 1;
      while (idEnd < end && validIdChar(*idEnd))
        ++idEnd;
      llvm::StringRef identifier(ptr + 1, (idEnd - (ptr + 1)));
      if (tryReplace(identifier, idEnd))
        continue;
    }
    if (*ptr == '$' && get(ptr + 1) == '{') {
      const char *idEnd = ptr + 2;
      while (idEnd < end && validIdChar(*idEnd))
        ++idEnd;
      if (get(idEnd) == '}') {
        llvm::StringRef identifier(ptr + 2, (idEnd - (ptr + 2)));
        if (tryReplace(identifier, idEnd + 1))
          continue;
      }
    }
    result.push_back(*(ptr++));
  }
  return result.str().str();
}
