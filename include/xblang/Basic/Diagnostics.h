//===- Diagnostics.h - XBLang Diagnostics ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares XBLang compiler diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_BASIC_DIAGNOSTICS_H
#define XBLANG_BASIC_DIAGNOSTICS_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "xblang/Basic/SourceLocation.h"

namespace xblang {
using ManagedDiagnostic = mlir::InFlightDiagnostic;
using mlir::Diagnostic;
using mlir::DiagnosticSeverity;
using mlir::emitError;
using mlir::emitRemark;
using mlir::emitWarning;

/// Creates an MLIR location from a SourceLocation
mlir::OpaqueLoc getLoc(mlir::MLIRContext *context, SourceLocation loc,
                       mlir::LocationAttr fallback = {});
} // namespace xblang

#endif // XBLANG_BASIC_DIAGNOSTICS_H
