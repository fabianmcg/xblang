#include "xblang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace xblang;

namespace {
template <typename T>
int log2i(T val) {
  return llvm::bit_width(static_cast<unsigned>(val)) - 1;
}

struct InvocationImpl : public CompilerInvocation {
  InvocationImpl() {
    static cl::OptionCategory cmdCat = cl::OptionCategory("xblang");
    cat = std::addressof(cmdCat);

    static cl::list<std::string, std::vector<std::string>> inputFilesCL(
        cl::Positional, cl::cat(cmdCat), cl::location(inputFiles),
        cl::desc("Input files"), cl::OneOrMore);

    static cl::opt<std::string, true> outputFileCL(
        "o", cl::cat(cmdCat), cl::location(outputFile),
        cl::desc("Specify output filename"), cl::value_desc("Output file"),
        cl::init("a.ll"));

    static cl::list<std::string, std::vector<std::string>> extensionsCL(
        "load-extension", cl::cat(cmdCat), cl::location(extensions),
        cl::desc("Language extensions"), cl::ZeroOrMore);

    static cl::opt<CompilerStage, true> compilationPipelineCL(
        "final-stage", cl::desc("Compilation stages:"), cl::cat(cmdCat),
        cl::location(finalStageToRun),
        cl::values(
            clEnumValN(RunAll, "all", "Run all stages."),
            clEnumValN(Parse, "parse", "Parse files and stop."),
            clEnumValN(Sema, "sema", "Run Sema and stop."),
            clEnumValN(CodeGen, "codegen", "Run CodeGen and stop."),
            clEnumValN(HighIRTransforms, "high-transforms",
                       "Apply high level MLIR transformations and stop."),
            clEnumValN(
                HighIRConcretization, "concretization",
                "Concretize the high level MLIR representation and stop."),
            clEnumValN(HighIRLowering, "high-lowering",
                       "Build the low level MLIR representation and stop."),
            clEnumValN(LowIRTransforms, "low-transforms",
                       "Apply low level MLIR transformations and stop."),
            clEnumValN(LowIRLowering, "llvm-lowering",
                       "Build the MLIR LLVM IR."),
            clEnumValN(LLVMIR, "llvm", "Build the LLVM IR.")),
        cl::init(RunAll));

    static cl::alias compilationPipelineCLAlias(
        "cc", cl::desc("Alias for -final-stage"),
        cl::aliasopt(compilationPipelineCL));

    static cl::opt<OptLevel, true> optLevelCL(
        cl::desc("Optimization level:"), cl::cat(cmdCat),
        cl::location(optLevel),
        cl::values(clEnumVal(O0, "Don't optimize."),
                   clEnumVal(O1, "Run with optimization level 1."),
                   clEnumVal(O2, "Run with optimization level 2."),
                   clEnumVal(O3, "Run with optimization level 3.")));

    static cl::opt<DebugLevel, true> debugLevelCL(
        cl::desc("Debug level:"), cl::cat(cmdCat), cl::location(debugLevel),
        cl::values(clEnumValN(G0, "g0", "Don't add debug information."),
                   clEnumValN(G1, "g1", "Run with debug level 1."),
                   clEnumValN(G2, "g2", "Run with debug level 2."),
                   clEnumValN(G3, "g3", "Run with debug level 3.")));

    static cl::bits<CompilerStage, unsigned> optPassesCL(
        cl::desc("MLIR optimization passes:"), cl::cat(cmdCat),
        cl::location(optPipeline),
        cl::values(
            clEnumValN(0, "opt-all",
                       "Run optimization passes on all MLIR representations."),
            clEnumValN(log2i(OptHighIR), "opt-high",
                       "Run passes on the high level MLIR representation."),
            clEnumValN(log2i(OptLowIR), "opt-low",
                       "Run passes on the low level MLIR representation."),
            clEnumValN(log2i(OptLLVM), "opt-llvm",
                       "Run passes on the LLVM MLIR representation.")));

    static cl::opt<bool, true> dumpReprCL(
        "dump", cl::cat(cmdCat), cl::location(dumpRepr),
        cl::desc("Dump the last representation available"));
  }

  cl::OptionCategory *cat{};
};

ManagedStatic<InvocationImpl> clInvocation;
} // namespace

llvm::cl::OptionCategory *CompilerInvocation::registerCLOpts() {
  return clInvocation->cat;
}

const CompilerInvocation &CompilerInvocation::getCLOpts() {
  auto &opts = *clInvocation;
  if ((opts.optPipeline & 1) == 1)
    opts.optPipeline = OptAll;
  return opts;
}
