//===- Util.h - Tablegen utilities -------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common tablegen utilities.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_TABLEGEN_UTIL_H
#define XBLANG_TABLEGEN_UTIL_H

#include <vector>

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace xblang {
namespace tablegen {
/// Returns whether the record belongs to the main file being processed by
/// tablegen.
bool isInMainFile(llvm::Record *record);

/// Sorts the RecordKeeper by order of appearance.
std::vector<llvm::Record *>
sortRecordDefinitions(const llvm::RecordKeeper &records);
std::vector<llvm::Record *>
sortRecordDefinitions(std::vector<llvm::Record *> &&records);
} // namespace tablegen
} // namespace xblang

#endif // XBLANG_TABLEGEN_UTIL_H
