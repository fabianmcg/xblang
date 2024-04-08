#include "xblang/Dialect/XBLang/IR/XBLang.h"

#include "mlir/IR/PatternMatch.h"
#include "xblang/Dialect/XBLang/IR/ASMUtils.h"
#include "xblang/Dialect/XBLang/IR/Dialect.h"
#include "xblang/Dialect/XBLang/IR/Type.h"

using namespace mlir;
using namespace xblang::xb;

#include "xblang/Dialect/XBLang/IR/XBLangDialect.cpp.inc"

void XBLangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xblang/Dialect/XBLang/IR/XBLang.cpp.inc"
      >();
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xblang/Dialect/XBLang/IR/XBLangAttributes.cpp.inc"
      >();
}

mlir::Operation *XBLangDialect::materializeConstant(mlir::OpBuilder &builder,
                                                    mlir::Attribute value,
                                                    mlir::Type type,
                                                    mlir::Location loc) {
  if (auto attr = dyn_cast<TypedAttr>(value))
    return builder.create<ConstantOp>(loc, type, attr);
  return nullptr;
}

namespace {
ParseResult parseConstant(OpAsmParser &parser, mlir::Attribute &valueAttr) {
  NamedAttrList attr;
  if (parser.parseAttribute(valueAttr, "value", attr).failed()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected constant attribute to match type");
  }
  return success();
}

void printConstant(OpAsmPrinter &p, ConstantOp op, Attribute value) {
  p.printAttribute(value);
}

ParseResult
parseRangeList(OpAsmParser &parser,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &vars,
               SmallVectorImpl<Type> &varTypes,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ranges,
               SmallVectorImpl<Type> &rangesTypes) {
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(vars.emplace_back()) ||
            parser.parseColonType(varTypes.emplace_back()) ||
            parser.parseKeyword("in") ||
            parser.parseOperand(ranges.emplace_back()) ||
            parser.parseColonType(rangesTypes.emplace_back()))
          return failure();
        return success();
      })))
    return failure();
  return success();
}

void printRangeList(OpAsmPrinter &p, Operation *op, OperandRange varOperands,
                    TypeRange varTypes, OperandRange ranges,
                    TypeRange rangeTypes) {
  for (unsigned i = 0, e = varOperands.size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << varOperands[i] << " : " << varOperands[i].getType() << " in ";
    p << ranges[i] << " : " << ranges[i].getType();
  }
}
} // namespace

#define GET_ATTRDEF_CLASSES
#include "xblang/Dialect/XBLang/IR/XBLangAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "xblang/Dialect/XBLang/IR/XBLang.cpp.inc"
