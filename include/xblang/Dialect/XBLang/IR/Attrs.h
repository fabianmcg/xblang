#ifndef XBLANG_DIALECT_XBLANG_IR_ATTRS_H
#define XBLANG_DIALECT_XBLANG_IR_ATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "xblang/Dialect/XBLang/IR/Enums.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "xblang/Dialect/XBLang/IR/XBLangAttributes.h.inc"

#endif
