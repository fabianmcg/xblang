//===- SourceManager.h - Source Manager --------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XBLang source manager.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_SOURCEMANAGER_H
#define XBLANG_BASIC_SOURCEMANAGER_H

#include "xblang/Basic/Diagnostics.h"
#include "xblang/Basic/SourceState.h"
#include "llvm/Support/FileSystem/UniqueID.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <memory>

namespace llvm {
class raw_ostream;
}

namespace mlir {
class MLIRContext;
}

namespace xblang {
class SourceManager;
struct SourceManagerImpl;

//===----------------------------------------------------------------------===//
// SMDiagnosticsEmitter
//===----------------------------------------------------------------------===//
/// Utility base class for emitting diagnostics from a SourceManager.
class SMDiagnosticsEmitter {
public:
  SMDiagnosticsEmitter(SourceManager *sourceManager)
      : sourceManager(sourceManager) {}

  /// Emit a diagnostic at location `loc`, with message `msg`.
  ManagedDiagnostic emitDiagnostic(SourceLocation loc,
                                   DiagnosticSeverity severity,
                                   llvm::Twine msg = "") const;

  /// Emit a diagnostic remark at location `loc`, with message `msg`.
  ManagedDiagnostic emitRemark(SourceLocation loc, llvm::Twine msg = "") const;

  /// Emit a diagnostic note at location `loc`, with message `msg`.
  ManagedDiagnostic emitWarning(SourceLocation loc, llvm::Twine msg = "") const;

  /// Emit a diagnostic note at location `loc`, with message `msg`.
  ManagedDiagnostic emitError(SourceLocation loc, llvm::Twine msg = "") const;

  /// Returns a source location with UnknownLoc or FileLineColLoc as fall-back
  /// location.
  mlir::OpaqueLoc getLoc(SourceLocation loc,
                         bool useFileLineColFallback = false) const;

  /// Returns a source location with unknown fall-back location.
  mlir::FileLineColLoc getFLCLoc(SourceLocation loc) const;

  /// Returns the source manager.
  SourceManager *getSourceManager() const { return sourceManager; }

protected:
  /// Source manager used to emit the diagnostics.
  SourceManager *sourceManager{};
};

//===----------------------------------------------------------------------===//
// Source
//===----------------------------------------------------------------------===//
/// This class represents a buffer to source code.
class Source : public SMDiagnosticsEmitter {
public:
  /// Returns source id in the source manager.
  int getId() const { return id; }

  /// Returns true if the source has a valid id.
  bool isValid() const { return id >= 0; }

  /// Returns a fresh source state.
  SourceState getState(int l = 1, int c = 1) const {
    return SourceState(buffer.getBuffer(), l, c);
  }

  /// Returns the filename.
  llvm::StringRef getFilename() const { return filename; }

  /// Returns the unique file system identifier.
  llvm::sys::fs::UniqueID getUID() const { return uniqueId; }

  /// Returns the source size.
  size_t size() const { return buffer.getBufferSize(); }

private:
  friend struct SourceManagerImpl;
  Source(SourceManager &owner, int id, llvm::MemoryBufferRef buffer,
         llvm::StringRef filename = {}, llvm::sys::fs::UniqueID uid = {});
  /// Reference to the memory buffer owned by the SourceManager.
  llvm::MemoryBufferRef buffer;
  /// Source file name.
  const std::string filename;
  /// UniqueID pointing to the loaded file.
  llvm::sys::fs::UniqueID uniqueId;
  /// Source id in the source manager.
  const int id = -1;
};

//===----------------------------------------------------------------------===//
// SourceManager
//===----------------------------------------------------------------------===//
/// Class for managing source files.
class SourceManager {
public:
  SourceManager(mlir::MLIRContext *context, llvm::raw_ostream *os = nullptr);
  ~SourceManager();
  /// Returns the source with identifier `id`.
  Source *get(size_t id) const;

  /// Loads a source file, returns nullptr if the file couldn't be loaded.
  Source *getSource(llvm::StringRef filename, SourceLocation loc = {});

  /// Creates a source file from a string.
  Source *createSource(llvm::StringRef contents,
                       llvm::StringRef identifier = {},
                       SourceLocation loc = {});

  /// Registers the source manager diagnostics handler in the MLIR context.
  void registerDiagnosticsHandler();

  /// Unregisters the source manager diagnostics handler in the MLIR context.
  void unregisterDiagnosticsHandler();

  /// Returns a source location with UnknownLoc or FileLineColLoc as fall-back
  /// location.
  mlir::OpaqueLoc getLoc(SourceLocation loc,
                         bool useFileLineColFallback = false) const;
  /// Returns a source location with unknown fall-back location.
  mlir::FileLineColLoc getFLCLoc(SourceLocation loc) const;

  /// Emit a diagnostic at location `loc`, with message `msg`.
  ManagedDiagnostic emitDiagnostic(SourceLocation loc,
                                   DiagnosticSeverity severity,
                                   llvm::Twine msg = "") const;

  /// Emit a diagnostic remark at location `loc`, with message `msg`.
  ManagedDiagnostic emitRemark(SourceLocation loc, llvm::Twine msg = "") const;

  /// Emit a diagnostic note at location `loc`, with message `msg`.
  ManagedDiagnostic emitWarning(SourceLocation loc, llvm::Twine msg = "") const;

  /// Emit a diagnostic note at location `loc`, with message `msg`.
  ManagedDiagnostic emitError(SourceLocation loc, llvm::Twine msg = "") const;

private:
  /// Pointer to the source manager implementation.
  const std::unique_ptr<SourceManagerImpl> impl;
};
} // namespace xblang

#endif // XBLANG_BASIC_SOURCEMANAGER_H
