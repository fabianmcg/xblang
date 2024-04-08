#ifndef XBLANG_DIALECT_XBLANG_IR_ASMUTILS_H
#define XBLANG_DIALECT_XBLANG_IR_ASMUTILS_H

#include "mlir/Support/LogicalResult.h"
#include "xblang/Support/LLVM.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {
class OpAsmParser;
class OpAsmPrinter;
class OpBuilder;
class Operation;
class Region;
} // namespace mlir

namespace xblang {
namespace xb {
using TerminatorBuilder = llvm::function_ref<void(OpBuilder &)>;
/// Creates a terminator on the last `Block` if it doesn't has one.
void ensureTerminator(Region &region, TerminatorBuilder terminatorBuilder);

/// Creates a `xb.yield Fallthrough` terminator if the region doesn't has a
/// terminator.
void maybeEnsureFallthroughYield(Region &region);

/// Creates a `xb.yield Fallthrough` terminator if the region doesn't has a
/// terminator.
mlir::ParseResult maybeEnsureFallthroughYield(OpAsmParser &parser,
                                              Region &region);

/// Parse a region with an implicit `xb.yield Fallthrough` terminator.
mlir::ParseResult parseRegionWithImplicitYield(OpAsmParser &parser,
                                               Region &region);

/// Print a region maybe omitting a `xb.yield Fallthrough` terminator.
void printRegionWithImplicitYield(OpAsmPrinter &printer, Operation *op,
                                  Region &region,
                                  bool printEntryBlockArgs = true,
                                  bool printEmptyBlock = false);
} // namespace xb
} // namespace xblang

#endif
