//===- Program.cpp - External programs driver --------------------*- C++-*-===//
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

#include "xblang/Support/Program.h"

using namespace xblang;

Program Program::make(StringRef program, llvm::ArrayRef<StringRef> paths) {
  llvm::ErrorOr<std::string> path =
      llvm::sys::findProgramByName(program, paths);
  Program prog(program, path ? std::move(path.get()) : "");
  if (!path) {
    prog.errorMessage = path.getError().message();
    prog.state = InvalidProgram;
  }
  return prog;
}

mlir::LogicalResult
Program::execute(llvm::ArrayRef<StringRef> args,
                 llvm::ArrayRef<std::optional<StringRef>> redirections,
                 std::optional<llvm::ArrayRef<StringRef>> env, bool async,
                 unsigned secondsToWait, unsigned memoryLimit) {
  if (state == InvalidProgram)
    return mlir::failure();
  if (state == Executing) {
    errorMessage =
        "the program hasn't finished, call wait before calling this function";
    return mlir::failure();
  }
  // Resets the internal state.
  reset();
  bool failed = false;
  // Execute.
  if (async)
    info = llvm::sys::ExecuteNoWait(programPath, args, env, redirections,
                                    memoryLimit, &errorMessage, &failed);
  else
    info.ReturnCode = llvm::sys::ExecuteAndWait(
        programPath, args, env, redirections, secondsToWait, memoryLimit,
        &errorMessage, &failed, &stats);
  if (info.ReturnCode != 0 || failed)
    state = Failed;
  if (async)
    state = Executing;
  return state == Failed ? mlir::failure() : mlir::success();
}

mlir::LogicalResult Program::wait(std::optional<unsigned> secondsToWait,
                                  bool polling) {
  if (state == Succeeded || state == Valid)
    return mlir::success();
  if (state != Executing)
    return mlir::failure();
  info = llvm::sys::Wait(info, secondsToWait, &errorMessage, &stats, polling);
  if (!polling && info.ReturnCode != 0)
    state = Failed;
  return mlir::success();
}

void Program::reset() {
  state = Valid;
  errorMessage.clear();
  stats = std::nullopt;
  info = llvm::sys::ProcessInfo();
  info.ReturnCode = 0;
}
