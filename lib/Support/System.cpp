//===- System.cpp - System utilities -----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines system utilities.
//
//===----------------------------------------------------------------------===//

#include "xblang/Support/System.h"

using namespace xblang;

//===----------------------------------------------------------------------===//
// BasicFile
//===----------------------------------------------------------------------===//

bool BasicFile::exists() const { return llvm::sys::fs::exists(getFilename()); }

std::unique_ptr<llvm::MemoryBuffer> BasicFile::load() {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(getFilename());
  if (!file) {
    error = file.getError();
    return {};
  }
  return std::move(file.get());
}

std::unique_ptr<llvm::raw_fd_ostream> BasicFile::open() {
  std::unique_ptr<llvm::raw_fd_ostream> stream(
      new llvm::raw_fd_ostream(getFilename(), error));
  return error ? nullptr : std::move(stream);
}

//===----------------------------------------------------------------------===//
// TempFile
//===----------------------------------------------------------------------===//

UniqueFile UniqueFile::make(const Twine &filename, bool erase) {
  UniqueFile file;
  filename.toVector(file.filename);
  file.remover.setFile(file.getFilename(), erase);
  return file;
}

UniqueFile UniqueFile::makeTemp(const Twine &prefix, StringRef suffix,
                                bool erase) {
  UniqueFile file;
  file.error =
      llvm::sys::fs::createTemporaryFile(prefix, suffix, file.filename);
  file.remover.setFile(file.getFilename(), erase);
  return file;
}
