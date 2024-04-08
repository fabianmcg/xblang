//===- xblang-tblgen.cpp - XBLang Tablegen Driver main ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main for xblang-tblgen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

#include "xblang/Support/Program.h"

using namespace mlir;
using namespace llvm;

// Generator to invoke.
static const mlir::GenInfo *generator;

namespace {
llvm::cl::OptionCategory clCat("General xblang-tablegen options");
}

static llvm::cl::opt<std::string>
    clangFormat("formatter", llvm::cl::desc("Path to clang-format"),
                llvm::cl::cat(clCat), llvm::cl::init("clang-format"));

static llvm::cl::opt<bool>
    formatOpt("format", llvm::cl::desc("Apply clang-format to the output"),
              llvm::cl::cat(clCat), llvm::cl::init(false));

// TableGenMain requires a function pointer so this function is passed in which
// simply wraps the call to the generator.
static bool xblangTableGenMain(raw_ostream &os, RecordKeeper &records) {
  if (!generator) {
    os << records;
    return false;
  }
  if (formatOpt) {
    auto prog = xblang::Program::make(clangFormat);
    if (prog.getState() != prog.Valid)
      return generator->invoke(records, os);
    if (auto outputFile = xblang::UniqueFile::makeTemp("tmp", "cpp")) {
      bool status = false;
      // Try to emit to a tmp file.
      if (auto outStream = outputFile.open()) {
        status = generator->invoke(records, *outStream);
      } else {
        llvm::errs() << "failed while opening the file with error: "
                     << outputFile.getError().message() << "\n";
        return generator->invoke(records, os);
      }
      // Try to invoke clang-format.
      if (failed(
              prog.execute({"clang-format", "-i", outputFile.getFilename()}))) {
        llvm::errs() << "`clang-format` execution failed with error: "
                     << prog.getError() << "\n";
        return true;
      }
      // Read back the buffer.
      if (auto buf = outputFile.load()) {
        os << buf->getBuffer();
      } else {
        llvm::errs() << "failed reading the buffer with error: "
                     << outputFile.getError().message() << "\n";
        return true;
      }
      return status;
    }
  }
  return generator->invoke(records, os);
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::opt<const mlir::GenInfo *, true, mlir::GenNameParser> generator(
      "", llvm::cl::desc("Generator to run"), cl::location(::generator));
  cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &xblangTableGenMain);
}
