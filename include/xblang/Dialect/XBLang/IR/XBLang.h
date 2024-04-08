#ifndef XBLANG_DIALECT_XBLANG_IR_XBLANG_H
#define XBLANG_DIALECT_XBLANG_IR_XBLANG_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "xblang/Dialect/XBLang/IR/Attrs.h"
#include "xblang/Dialect/XBLang/IR/Dialect.h"
#include "xblang/Dialect/XBLang/IR/Interfaces.h"
#include "xblang/Dialect/XBLang/IR/Type.h"

namespace mlir {
class PatternRewriter;
}

#define GET_OP_CLASSES
#include "xblang/Dialect/XBLang/IR/XBLang.h.inc"

#endif
