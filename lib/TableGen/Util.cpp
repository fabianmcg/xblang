//===- Util.h - Tablegen utilities -------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common tablegen utilities.
//
//===----------------------------------------------------------------------===//

#include "xblang/TableGen/Util.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace xblang::tablegen;

bool xblang::tablegen::isInMainFile(llvm::Record *record) {
  if (!record)
    return false;
  auto locs = record->getLoc();
  if (!record->isAnonymous() && locs.size() &&
      llvm::SrcMgr.getMainFileID() ==
          llvm::SrcMgr.FindBufferContainingLoc(locs[0]))
    return true;
  return false;
}

std::vector<llvm::Record *>
xblang::tablegen::sortRecordDefinitions(const llvm::RecordKeeper &records) {
  auto &defs = records.getDefs();
  std::vector<llvm::Record *> definitions;
  definitions.reserve(defs.size());
  for (auto &kv : defs)
    definitions.push_back(kv.second.get());
  return sortRecordDefinitions(std::move(definitions));
}

std::vector<llvm::Record *>
xblang::tablegen::sortRecordDefinitions(std::vector<llvm::Record *> &&records) {
  std::sort(records.begin(), records.end(),
            [](llvm::Record *lhs, llvm::Record *rhs) {
              return lhs->getID() < rhs->getID();
            });
  return std::move(records);
}
