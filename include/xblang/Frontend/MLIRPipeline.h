#ifndef XBLANG_FRONTEND_MLIRPIPELINE_H
#define XBLANG_FRONTEND_MLIRPIPELINE_H

#include <memory>

namespace mlir {
class PassManager;
}

namespace xblang {
class CompilerInstance;

class MLIRPipeline {
public:
  MLIRPipeline(CompilerInstance &ci);
  ~MLIRPipeline();
  int generateLLVM();
  int run();
  CompilerInstance &ci;
  std::unique_ptr<mlir::PassManager> pm;
};
} // namespace xblang
#endif
