#ifndef XBLANG_DIALECT_XBLANG_CONCRETIZATION_CONCRETIZATION_H
#define XBLANG_DIALECT_XBLANG_CONCRETIZATION_CONCRETIZATION_H

namespace mlir {
class PassManager;
class RewritePatternSet;
} // namespace mlir

namespace xblang {
namespace xb {
class XBLangTypeSystem;
void populateConcretizationPasses(XBLangTypeSystem &typeSystem,
                                  mlir::PassManager &pm);
void populateConcretizationPatterns(XBLangTypeSystem &typeSystem,
                                    mlir::RewritePatternSet &patterns);
} // namespace xb
} // namespace xblang

#endif
