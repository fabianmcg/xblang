#ifndef XBLANG_DIALECT_PARALLEL_TRANSFORMS_PASSES_H
#define XBLANG_DIALECT_PARALLEL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "xblang/Lang/Parallel/Frontend/Options.h"

#include <memory>

namespace mlir {
class TypeConverter;
class RewritePatternSet;
class ConversionTarget;

namespace par {
using ::xblang::par::ParOptions;

#define GEN_PASS_DECL
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"

void populateTransformationPasses(mlir::PassManager &pm,
                                  ParOptions options = {});
void populateConcretizationPasses(mlir::PassManager &pm,
                                  ParOptions options = {});
void populateLoweringTransformsPasses(mlir::PassManager &pm,
                                      ParOptions options = {});
void populateLLVMLoweringPasses(mlir::PassManager &pm, ParOptions options = {});

void populateTransformationPatterns(RewritePatternSet &patterns);
void populateConcretizationPatterns(RewritePatternSet &patterns);
void populateLoweringPatterns(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &converter);
} // namespace par
} // namespace mlir

#endif
