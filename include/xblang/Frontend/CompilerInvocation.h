#ifndef XBLANG_FRONTEND_COMPILERINVOCATION_H
#define XBLANG_FRONTEND_COMPILERINVOCATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace llvm {
namespace cl {
class OptionCategory;
}
} // namespace llvm

namespace xblang {
class CompilerInvocation {
public:
  typedef enum : unsigned {
    Parse,
    Sema,
    CodeGen,
    HighIRTransforms,
    HighIRConcretization,
    HighIRLowering,
    LowIRTransforms,
    LowIRLowering,
    LLVMIR,

    MainStages = 31,
    OptHighIR = 32,
    OptLowIR = 64,
    OptLLVM = 128,
    OptAll = OptHighIR | OptLowIR | OptLLVM,

    RunAll = LLVMIR | OptAll,
  } CompilerStage;

  typedef enum : unsigned { O0, O1, O2, O3 } OptLevel;

  typedef enum : unsigned { G0, G1, G2, G3 } DebugLevel;
  friend class Driver;
  CompilerInvocation() = default;
  CompilerInvocation(const CompilerInvocation &) = default;
  CompilerInvocation &operator=(const CompilerInvocation &) = default;
  CompilerInvocation(CompilerInvocation &&) = default;
  CompilerInvocation &operator=(CompilerInvocation &&) = default;

  llvm::StringRef getOutputFile() const { return outputFile; }

  llvm::ArrayRef<std::string> getInputFiles() const {
    return llvm::ArrayRef<std::string>(inputFiles);
  }

  llvm::ArrayRef<std::string> getExtensions() const { return extensions; }

  const std::vector<std::string> &getIncludeDirs() const { return includeDirs; }

  unsigned getCompilationPipeline() const {
    return finalStageToRun | optPipeline;
  }

  unsigned mainFinalStage() const {
    return getCompilationPipeline() & MainStages;
  }

  bool isBetweenStages(CompilerStage lower, CompilerStage upper) const {
    auto main = mainFinalStage();
    auto l = lower & MainStages;
    auto u = upper & MainStages;
    return (l <= main) && (main <= u);
  }

  bool isFinalStage(CompilerStage stage) const {
    return mainFinalStage() == stage;
  }

  OptLevel getOptLevel() const { return optLevel; }

  DebugLevel getDebugLevel() const { return debugLevel; }

  bool runStage(CompilerStage stage) const { return mainFinalStage() >= stage; }

  bool runOptStage(CompilerStage stage) const {
    return ((getCompilationPipeline() & ~MainStages) & stage) == stage;
  }

  bool dump() const { return dumpRepr; }

  static const CompilerInvocation &getCLOpts();
  static llvm::cl::OptionCategory *registerCLOpts();

protected:
  std::string outputFile{};
  std::vector<std::string> inputFiles{};
  std::vector<std::string> includeDirs{};
  std::vector<std::string> extensions{};
  CompilerStage finalStageToRun = LLVMIR;
  unsigned optPipeline = 0;
  OptLevel optLevel{O0};
  DebugLevel debugLevel{G0};
  bool dumpRepr{false};
};
} // namespace xblang

#endif
