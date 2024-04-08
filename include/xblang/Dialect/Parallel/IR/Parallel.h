#ifndef XBLANG_DIALECT_PARALLEL_IR_PARALLEL_H
#define XBLANG_DIALECT_PARALLEL_IR_PARALLEL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "xblang/Dialect/Parallel/IR/Enums.h"
#include "xblang/Dialect/XBLang/IR/ASMUtils.h"
#include "xblang/Dialect/XBLang/IR/Interfaces.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace par {
using xblang::xb::parseRegionWithImplicitYield;
using xblang::xb::printRegionWithImplicitYield;

/// Class holding loop information.
class LoopInfo {
public:
  LoopInfo() = default;
  LoopInfo(StringAttr varName, LoopComparatorOp cmpOp = LoopComparatorOp::Less,
           LoopStepOp stepOp = LoopStepOp::Add);

  /// Returns true if all members are the same.
  bool operator==(const LoopInfo &info) const;

  /// Variable name to be used in ODS.
  StringAttr varName{};

  /// Compare operation.
  LoopComparatorOp cmpOp = LoopComparatorOp::Less;

  /// Step operation.
  LoopStepOp stepOp = LoopStepOp::Add;
};

/// Hash a LoopInfo value
llvm::hash_code hash_value(const mlir::par::LoopInfo &value);

/// Convert `value` to a DictAttr.
Attribute convertToAttribute(MLIRContext *ctx,
                             const SmallVector<par::LoopInfo> &value);

/// Convert `attr` from a DictAttr.
LogicalResult
convertFromAttribute(SmallVector<par::LoopInfo> &value, Attribute attr,
                     function_ref<InFlightDiagnostic()> diagnostic);

/// Helper class for building loops.
struct LoopBuilder {
  LoopBuilder(StringRef varName, Value begin, Value end,
              std::optional<Location> varLoc = std::nullopt,
              LoopComparatorOp cmpOp = LoopComparatorOp::Less, Value step = {},
              LoopStepOp stepOp = LoopStepOp::Add);
  /// Variable name to be used in ODS.
  std::string varName;
  /// Location of the variable.
  std::optional<Location> varLoc;
  /// Begin of the loop.
  Value begin;
  /// End of the loop.
  Value end;
  /// Step of the loop, by default `1`.
  Value step{};
  /// Loop comparator.
  LoopComparatorOp cmpOp = LoopComparatorOp::Less;
  /// Step operator.
  LoopStepOp stepOp = LoopStepOp::Add;
};

/// Class describing a single nested loop.
struct LoopDescriptor {
  /// Iteration variable.
  Value var;
  /// Begin of the loop.
  Value begin;
  /// End of the loop.
  Value end;
  /// Step of the loop.
  Value step;
  /// Loop information.
  LoopInfo info;
};

} // namespace par
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "xblang/Dialect/Parallel/IR/ParallelTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "xblang/Dialect/Parallel/IR/ParallelAttributes.h.inc"

#define GET_OP_CLASSES
#include "xblang/Dialect/Parallel/IR/Parallel.h.inc"

#endif
