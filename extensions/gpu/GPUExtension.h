//===- GPUExtension.h - Declares the gpu extension ---------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the gpu extension.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_EXTENSION_H
#define GPU_EXTENSION_H

#include "gpu/InitExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "xblang/Lang/XBLang/Syntax/Lexer.h"
#include "xblang/Syntax/ParserBase.h"
#include "xblang/XLG/Builder.h"

#include "gpu/GPUParser.h.inc"

#endif // GPU_EXTENSION_H
