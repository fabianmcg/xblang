#ifndef XBLANG_SUPPORT_LLVM_H
#define XBLANG_SUPPORT_LLVM_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class ArrayAttr;
class Attribute;
class Block;
class ComplexType;
class ConversionPattern;
class ConversionTarget;
class Dialect;
class DialectAsmParser;
class DialectAsmPrinter;
class DictionaryAttr;
class FloatType;
class FrozenRewritePatternSet;
class FunctionType;
class IndexType;
class IntegerAttr;
class IntegerType;
class Location;
class LocationAttr;
class MemRefType;
class MLIRContext;
class NamedAttribute;
class OpAsmParser;
class OpAsmPrinter;
class OpBuilder;
class Operation;
class OpOperand;
class Pass;
class PassManager;
class Pattern;
class PatternBenefit;
class PatternRewriter;
class RankedTensorType;
class Region;
class RewritePatternSet;
class StringAttr;
class TensorType;
class Type;
class TypeAttr;
class TypeConverter;
class TypedAttr;
class TypeID;
class TypeRange;
class Value;
class ValueRange;
struct LogicalResult;

LogicalResult success(bool isSuccess);
LogicalResult failure(bool isFailure);
bool succeeded(LogicalResult result);
bool failed(LogicalResult result);
} // namespace mlir

namespace xblang {
// Casting operators.
using llvm::cast;
using llvm::cast_if_present;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_if_present;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::isa_and_nonnull;
using llvm::isa_and_present;

// String types
using llvm::SmallString;
using llvm::StringLiteral;
using llvm::StringRef;
using llvm::Twine;

// Container Related types
//
// Containers.
using llvm::ArrayRef;
using llvm::BitVector;
template <typename T, typename Enable = void>
using DenseMapInfo = llvm::DenseMapInfo<T, Enable>;
template <typename KeyT, typename ValueT,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename BucketT = llvm::detail::DenseMapPair<KeyT, ValueT>>
using DenseMap = llvm::DenseMap<KeyT, ValueT, KeyInfoT, BucketT>;
template <typename ValueT, typename ValueInfoT = DenseMapInfo<ValueT>>
using DenseSet = llvm::DenseSet<ValueT, ValueInfoT>;
template <typename AllocatorTy = llvm::MallocAllocator>
using StringSet = llvm::StringSet<AllocatorTy>;
using llvm::MutableArrayRef;
using llvm::PointerUnion;
using llvm::SmallPtrSet;
using llvm::SmallPtrSetImpl;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
template <typename T, typename R = T>
using StringSwitch = llvm::StringSwitch<T, R>;
using llvm::TinyPtrVector;
template <typename T, typename ResultT = void>
using TypeSwitch = llvm::TypeSwitch<T, ResultT>;

// Other common classes.
using llvm::APFloat;
using llvm::APInt;
using llvm::APSInt;
template <typename Fn>
using function_ref = llvm::function_ref<Fn>;
using llvm::iterator_range;
using llvm::raw_ostream;
using llvm::SMLoc;
using llvm::SMRange;

// MLIR classes
using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::Block;
using mlir::ComplexType;
using mlir::ConversionPattern;
using mlir::ConversionTarget;
using mlir::Dialect;
using mlir::DialectAsmParser;
using mlir::DialectAsmPrinter;
using mlir::DictionaryAttr;
using mlir::FloatType;
using mlir::FrozenRewritePatternSet;
using mlir::FunctionType;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
using mlir::LocationAttr;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpOperand;
using mlir::Pass;
using mlir::PassManager;
using mlir::Pattern;
using mlir::PatternBenefit;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::RewritePatternSet;
using mlir::StringAttr;
using mlir::TensorType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::TypeConverter;
using mlir::TypedAttr;
using mlir::TypeID;
using mlir::TypeRange;
using mlir::Value;
using mlir::ValueRange;

using mlir::failed;
using mlir::failure;
using mlir::succeeded;
using mlir::success;
} // namespace xblang

#endif
