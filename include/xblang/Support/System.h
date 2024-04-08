//===- System.h - System utilities -------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares system utilities.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SUPPORT_SYSTEM_H
#define XBLANG_SUPPORT_SYSTEM_H

#include "xblang/Support/LLVM.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"

namespace xblang {
/// Class for interacting with files.
class BasicFile {
public:
  BasicFile(const Twine &filename) { filename.toVector(this->filename); }

  operator bool() const { return error.value() == 0; }

  /// Returns whether the file exists.
  bool exists() const;

  /// Loads the contents of a file.
  std::unique_ptr<llvm::MemoryBuffer> load();

  /// Opens a file for writing.
  std::unique_ptr<llvm::raw_fd_ostream> open();

  /// Returns the filename.
  StringRef getFilename() const { return filename; }

  /// Returns the current error code.
  std::error_code getError() const { return error; }

protected:
  BasicFile() {}

  llvm::SmallString<64> filename;
  std::error_code error;
};

/// Class for automatically managing files, erasing them on destruction.
class UniqueFile : public BasicFile {
public:
  UniqueFile(UniqueFile &&) = default;
  UniqueFile(const UniqueFile &) = delete;
  UniqueFile &operator=(UniqueFile &&) = default;
  UniqueFile &operator=(const UniqueFile &) = delete;
  /// Creates a new unique file.
  static UniqueFile make(const Twine &filename, bool erase = true);
  /// Creates a temporary file.
  static UniqueFile makeTemp(const Twine &prefix = "tmp", StringRef suffix = "",
                             bool erase = true);

  /// Releases the ownership of the file.
  void release() { remover.releaseFile(); }

private:
  UniqueFile() : BasicFile() {}

  llvm::FileRemover remover;
};
} // namespace xblang

#endif // XBLANG_SUPPORT_SYSTEM_H
