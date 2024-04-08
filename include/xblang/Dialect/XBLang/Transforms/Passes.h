#ifndef XBLANG_DIALECT_XBLANG_TRANSFORMS_PASSES_H
#define XBLANG_DIALECT_XBLANG_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace xblang {
namespace xb {
class XBLangTypeConverter;

void populateXblangToStd(mlir::ConversionTarget &conversionTarget,
                         mlir::RewritePatternSet &patterns,
                         const XBLangTypeConverter &typeConverter);

void populateXBLangToCF(mlir::TypeConverter &, mlir::RewritePatternSet &);

void populateXBLangToLLVMConversionPatterns(mlir::TypeConverter &,
                                            mlir::RewritePatternSet &);

void populateXBLangGPUToLLVMConversionPatterns(mlir::TypeConverter &,
                                               mlir::RewritePatternSet &);

void populateXBLangGPUToSPIRVConversionPatterns(mlir::TypeConverter &,
                                                mlir::RewritePatternSet &);

#define GEN_PASS_DECL
#include "xblang/Dialect/XBLang/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "xblang/Dialect/XBLang/Transforms/Passes.h.inc"
} // namespace xb
} // namespace xblang

#endif
