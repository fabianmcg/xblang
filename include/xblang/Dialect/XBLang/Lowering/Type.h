#ifndef XBLANG_DIALECT_XBLANG_LOWERING_TYPE_H
#define XBLANG_DIALECT_XBLANG_LOWERING_TYPE_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "xblang/Dialect/XBLang/IR/Type.h"

namespace xblang {
namespace xb {
class XBLangTypeConverter : public TypeConverter {

public:
  XBLangTypeConverter(MLIRContext &context);

  /// Returns the MLIR context.
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context{};
};

class XBLangToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  XBLangToLLVMTypeConverter(MLIRContext *ctx,
                            const mlir::DataLayoutAnalysis *analysis = nullptr);
  XBLangToLLVMTypeConverter(MLIRContext *ctx,
                            const mlir::LowerToLLVMOptions &options,
                            const mlir::DataLayoutAnalysis *analysis = nullptr);

protected:
  void init();
  Type convertReferenceType(ReferenceType type) const;
  Type convertPointerType(PointerType type) const;
  Type convertAddressType(AddressType type) const;
  Type convertStructType(StructType type) const;
  Type convertNamedType(NamedType type) const;
};
} // namespace xb
} // namespace xblang

#endif
