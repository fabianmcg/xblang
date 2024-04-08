//===- SourceManager.cpp - Source Manager ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the source manager.
//
//===----------------------------------------------------------------------===//

#include "xblang/Basic/SourceManager.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"

using namespace xblang;

mlir::OpaqueLoc xblang::getLoc(mlir::MLIRContext *context, SourceLocation loc,
                               mlir::LocationAttr fallback) {
  if (fallback)
    return mlir::OpaqueLoc::get(loc.getLoc(), fallback);
  return mlir::OpaqueLoc::get(loc.getLoc(), context);
}

//===----------------------------------------------------------------------===//
// SMDiagnosticsEmitter
//===----------------------------------------------------------------------===//

ManagedDiagnostic SMDiagnosticsEmitter::emitDiagnostic(
    SourceLocation loc, DiagnosticSeverity severity, llvm::Twine msg) const {
  return sourceManager ? sourceManager->emitDiagnostic(loc, severity, msg)
                       : ManagedDiagnostic();
}

ManagedDiagnostic SMDiagnosticsEmitter::emitRemark(SourceLocation loc,
                                                   llvm::Twine msg) const {
  return sourceManager ? sourceManager->emitRemark(loc, msg)
                       : ManagedDiagnostic();
}

ManagedDiagnostic SMDiagnosticsEmitter::emitWarning(SourceLocation loc,
                                                    llvm::Twine msg) const {
  return sourceManager ? sourceManager->emitWarning(loc, msg)
                       : ManagedDiagnostic();
}

ManagedDiagnostic SMDiagnosticsEmitter::emitError(SourceLocation loc,
                                                  llvm::Twine msg) const {
  return sourceManager ? sourceManager->emitError(loc, msg)
                       : ManagedDiagnostic();
}

mlir::OpaqueLoc
SMDiagnosticsEmitter::getLoc(SourceLocation loc,
                             bool useFileLineColFallback) const {
  return sourceManager ? sourceManager->getLoc(loc, useFileLineColFallback)
                       : mlir::OpaqueLoc();
}

mlir::FileLineColLoc SMDiagnosticsEmitter::getFLCLoc(SourceLocation loc) const {
  return sourceManager ? sourceManager->getFLCLoc(loc) : mlir::FileLineColLoc();
}

//===----------------------------------------------------------------------===//
// Source
//===----------------------------------------------------------------------===//

Source::Source(SourceManager &sourceManager, int id,
               llvm::MemoryBufferRef buffer, llvm::StringRef filename,
               llvm::sys::fs::UniqueID uid)
    : SMDiagnosticsEmitter(&sourceManager), buffer(buffer), filename(filename),
      uniqueId(uid), id(id) {}

//===----------------------------------------------------------------------===//
// SourceManagerImpl implementation
//===----------------------------------------------------------------------===//
struct xblang::SourceManagerImpl {
  SourceManagerImpl(mlir::MLIRContext *context, llvm::raw_ostream *os)
      : context(context), os(os) {
    if (!this->os)
      this->os = &llvm::errs();
  }

  // Get or load a source file.
  Source *getSource(SourceManager *owner, llvm::StringRef filename,
                    SourceLocation loc);
  // Create a source file from a string.
  Source *create(SourceManager *owner, llvm::StringRef contents,
                 llvm::StringRef identifier, SourceLocation loc);
  // If loc belongs to the source manager, it returns the `FileLineColLoc` for
  // loc, nullptr otherwise.
  mlir::FileLineColLoc getLoc(SourceLocation loc);
  // Try to report a diagnostic.
  mlir::LogicalResult report(mlir::Diagnostic &idagnostic);
  // MLIR context used for diagnostics.
  mlir::MLIRContext *context;
  // Stream used to print diagnostics.
  llvm::raw_ostream *os;
  // LLVM source manager for diagnostics.
  llvm::SourceMgr manager;
  // Sources being managed.
  llvm::SmallVector<std::pair<Source, unsigned>> sources;
  // Lookup table from UniqueID to sources.
  llvm::DenseMap<llvm::sys::fs::UniqueID, Source *> sourcesLT;
  // MLIR diagnostic handler ID.
  mlir::DiagnosticEngine::HandlerID diagHandlerID{};
};

Source *SourceManagerImpl::getSource(SourceManager *owner,
                                     llvm::StringRef filename,
                                     SourceLocation loc) {
  llvm::sys::fs::UniqueID uid;
  if (std::error_code ec = llvm::sys::fs::getUniqueID(filename, uid)) {
    llvm::errs() << "could not get the file's unique id, the error code was: "
                 << ec.message() << "\n";
    return nullptr;
  }
  Source *source = sourcesLT[uid];
  if (source)
    return source;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = buffer.getError()) {
    llvm::errs() << "could not open the input file, the error code was: "
                 << ec.message() << "\n";
    return nullptr;
  }
  llvm::MemoryBufferRef bufRef = buffer.get()->getMemBufferRef();
  unsigned id = manager.AddNewSourceBuffer(std::move(buffer.get()), loc.loc);
  sources.push_back(
      {Source(*owner, sources.size(), bufRef, filename, uid), id});
  return (sourcesLT[uid] = &sources.back().first);
}

Source *SourceManagerImpl::create(SourceManager *owner,
                                  llvm::StringRef contents,
                                  llvm::StringRef identifier,
                                  SourceLocation loc) {
  auto buffer = llvm::MemoryBuffer::getMemBuffer(contents, identifier);
  llvm::MemoryBufferRef bufRef = buffer->getMemBufferRef();
  unsigned id = manager.AddNewSourceBuffer(std::move(buffer), loc.loc);
  sources.push_back({Source(*owner, sources.size(), bufRef), id});
  return &sources.back().first;
}

namespace {
llvm::SourceMgr::DiagKind getDiagKind(mlir::DiagnosticSeverity kind) {
  switch (kind) {
  case mlir::DiagnosticSeverity::Note:
    return llvm::SourceMgr::DK_Note;
  case mlir::DiagnosticSeverity::Warning:
    return llvm::SourceMgr::DK_Warning;
  case mlir::DiagnosticSeverity::Error:
    return llvm::SourceMgr::DK_Error;
  case mlir::DiagnosticSeverity::Remark:
    return llvm::SourceMgr::DK_Remark;
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}
} // namespace

mlir::LogicalResult SourceManagerImpl::report(mlir::Diagnostic &diagnostic) {
  auto loc = mlir::dyn_cast<mlir::OpaqueLoc>(diagnostic.getLocation());
  if (!loc)
    return mlir::failure();
  auto smLoc = llvm::SMLoc::getFromPointer(
      reinterpret_cast<const char *>(loc.getUnderlyingLocation()));
  if (!manager.FindBufferContainingLoc(smLoc))
    return mlir::failure();
  llvm::SMDiagnostic diag = manager.GetMessage(
      smLoc, getDiagKind(diagnostic.getSeverity()), diagnostic.str());
  manager.PrintMessage(*os, smLoc, getDiagKind(diagnostic.getSeverity()),
                       diagnostic.str());
  return mlir::success();
}

mlir::FileLineColLoc SourceManagerImpl::getLoc(SourceLocation loc) {
  assert(context && "invalid MLIR context");
  unsigned id = manager.FindBufferContainingLoc(loc.loc);
  if (id == 0)
    return nullptr;
  return mlir::FileLineColLoc::get(
      context, manager.getMemoryBuffer(id)->getBufferIdentifier(), loc.line,
      loc.column);
}

//===----------------------------------------------------------------------===//
// SourceManager implementation
//===----------------------------------------------------------------------===//

SourceManager::SourceManager(mlir::MLIRContext *context, llvm::raw_ostream *os)
    : impl(new SourceManagerImpl(context, os)) {}

SourceManager::~SourceManager() = default;

Source *SourceManager::get(size_t id) const {
  if (id < impl->sources.size())
    return &(impl->sources[id].first);
  return nullptr;
}

Source *SourceManager::getSource(llvm::StringRef filename, SourceLocation loc) {
  return impl->getSource(this, filename, loc);
}

Source *SourceManager::createSource(llvm::StringRef contents,
                                    llvm::StringRef identifier,
                                    SourceLocation loc) {
  return impl->create(this, contents, identifier, loc);
}

void SourceManager::registerDiagnosticsHandler() {
  if (!impl->context)
    return;
  SourceManagerImpl *impl = this->impl.get();
  impl->diagHandlerID = impl->context->getDiagEngine().registerHandler(
      [impl](mlir::Diagnostic &diagnostic) -> mlir::LogicalResult {
        return impl->report(diagnostic);
      });
}

void SourceManager::unregisterDiagnosticsHandler() {
  if (!impl->context)
    return;
  impl->context->getDiagEngine().eraseHandler(impl->diagHandlerID);
  impl->diagHandlerID = 0;
}

mlir::OpaqueLoc SourceManager::getLoc(SourceLocation loc,
                                      bool useFileLineColFallback) const {
  if (!impl->context)
    return nullptr;
  return xblang::getLoc(impl->context, loc,
                        useFileLineColFallback ? impl->getLoc(loc) : nullptr);
}

mlir::FileLineColLoc SourceManager::getFLCLoc(SourceLocation loc) const {
  if (!impl->context)
    return nullptr;
  return impl->getLoc(loc);
}

ManagedDiagnostic SourceManager::emitDiagnostic(SourceLocation loc,
                                                DiagnosticSeverity severity,
                                                llvm::Twine msg) const {
  if (!impl->context)
    return {};
  return impl->context->getDiagEngine().emit(getFLCLoc(loc), severity) << msg;
}

ManagedDiagnostic SourceManager::emitRemark(SourceLocation loc,
                                            llvm::Twine msg) const {
  if (!impl->context)
    return {};
  return xblang::emitRemark(getLoc(loc), msg);
}

ManagedDiagnostic SourceManager::emitWarning(SourceLocation loc,
                                             llvm::Twine msg) const {
  if (!impl->context)
    return {};
  return xblang::emitWarning(getLoc(loc), msg);
}

ManagedDiagnostic SourceManager::emitError(SourceLocation loc,
                                           llvm::Twine msg) const {
  if (!impl->context)
    return {};
  return xblang::emitError(getLoc(loc), msg);
}
