//===- Program.h - External programs driver ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares classes for interacting with external programs.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_SUPPORT_TOOLS_H
#define XBLANG_SUPPORT_TOOLS_H

#include "mlir/Support/LogicalResult.h"
#include "xblang/Support/LLVM.h"
#include "xblang/Support/System.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Program.h"

namespace xblang {
/// Class for executing external programs.
class Program {
public:
  typedef enum {
    InvalidArgs = -3,
    Failed = -2,
    InvalidProgram = -1,
    Valid = 0,
    Executing,
    Succeeded,
  } State;

  ~Program() {
    // Kill the process.
    (void)wait(0);
  }

  static Program make(StringRef program, llvm::ArrayRef<StringRef> paths = {});

  /// Executes a program.
  mlir::LogicalResult
  execute(llvm::ArrayRef<StringRef> args = {},
          llvm::ArrayRef<std::optional<StringRef>> redirections = {},
          std::optional<llvm::ArrayRef<StringRef>> env = std::nullopt,
          bool async = false, unsigned secondsToWait = 0,
          unsigned memoryLimit = 0);

  /// Waits for the program to finish execution.
  mlir::LogicalResult wait(std::optional<unsigned> secondsToWait = std::nullopt,
                           bool polling = false);

  /// Returns the program.
  StringRef getProgram() const { return program; }

  /// Returns the program path.
  StringRef getProgramPath() const { return programPath; }

  /// Returns the error message if there is one.
  StringRef getError() const { return errorMessage; }

  /// Gets the return code returned by the program.
  int getReturnCode() const { return info.ReturnCode; }

  /// Returns the process statistics.
  std::optional<llvm::sys::ProcessStatistics> getStats() const { return stats; }

  /// Returns the state of execution.
  State getState() { return state; }

private:
  /// Resets the internal state.
  void reset();

  Program(StringRef program, std::string &&path)
      : program(program.str()), programPath(std::move(path)) {}

  std::string program;
  std::string programPath;
  std::string errorMessage = {};
  llvm::sys::ProcessInfo info = {};
  std::optional<llvm::sys::ProcessStatistics> stats = std::nullopt;
  State state = Valid;
};
} // namespace xblang

#endif // XBLANG_SUPPORT_TOOLS_H
