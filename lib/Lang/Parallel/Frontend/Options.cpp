#include "xblang/Lang/Parallel/Frontend/Options.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

#include <vector>

using namespace xblang;
using namespace xblang::par;

using ParBackend = ParOptions::ParBackend;

namespace xblang {
namespace par {
struct ParOptionsImpl {
  typedef enum : int { O0, O1, O2, O3 } OptLevel;

  std::string targetTriple;
  std::string targetChip;
  std::string targetFeatures;
  std::string toolkitPath;
  std::string toolOpts;
  SmallVector<std::string, 0> linkLibs;
  OptLevel optimizationLevel;
  ParBackend parBackend;
  bool emitLLVM;
};
} // namespace par
} // namespace xblang

namespace {
struct ParOptionsCL {
  ParOptionsCL() {
    using namespace llvm::cl;
    static OptionCategory cmdCat = OptionCategory("par");
    this->cmdCat = std::addressof(cmdCat);

    auto setOpt = [](opt<std::string, true> &option, StringRef value) {
      if (option.getValue().empty())
        option.setValue(value.str(), true);
    };

    static opt<std::string, true> targetTriple(
        cat(cmdCat), "offload-triple", desc("Target triple:"),
        location(options.targetTriple), init(""));

    static opt<std::string, true> targetChip(
        cat(cmdCat), "offload-chip", desc("Target chip:"),
        location(options.targetChip), init(""));

    static opt<std::string, true> targetFeatures(
        cat(cmdCat), "offload-features", desc("Target features:"),
        location(options.targetFeatures), init(""));

    static opt<std::string, true> toolkitPath(
        cat(cmdCat), "offload-toolkit", desc("Toolkit path:"),
        location(options.toolkitPath), init(""));

    static opt<std::string, true> toolOpts(
        cat(cmdCat), "offload-opts", desc("Tool options:"),
        location(options.toolOpts), init(""));

    static list<std::string, SmallVector<std::string, 0>> linkLibs(
        cat(cmdCat), "offload-libs", location(options.linkLibs),
        desc("Offload link libraries."), ZeroOrMore);

    static opt<ParOptionsImpl::OptLevel, true> optimizationLevel(
        cat(cmdCat), "offload-opt", desc("Optimization level:"),
        location(options.optimizationLevel),
        values(clEnumValN(0, "O0", "No optimizations, enable debugging"),
               clEnumValN(1, "O1", "Enable trivial optimizations"),
               clEnumValN(2, "O2", "Enable default optimizations"),
               clEnumValN(3, "O3", "Enable expensive optimizations")),
        init(ParOptionsImpl::O3));

    static opt<ParBackend, true> parBackend(
        cat(cmdCat), "par", desc("Parallelization back-end:"),
        values(clEnumValN(ParBackend::seq, "seq", "Sequential code."),
               clEnumValN(ParBackend::host, "mp", "Multi-core parallelism."),
               clEnumValN(ParBackend::nvptx, "nvptx",
                          "NVPTX offload parallelism."),
               clEnumValN(ParBackend::amdgpu, "amdgpu",
                          "AMDGPU offload parallelism."),
               clEnumValN(ParBackend::spirv, "spirv",
                          "SYCL offload parallelism.")),
        location(options.parBackend), init(ParBackend::seq),
        callback([&](const uint8_t &backend) {
          switch (static_cast<ParBackend>(backend)) {
          case ParBackend::amdgpu:
            setOpt(targetTriple, "amdgcn-amd-amdhsa");
            setOpt(targetChip, "gfx90a");
            break;
          case ParBackend::nvptx:
            setOpt(targetTriple, "nvptx64-nvidia-cuda");
            setOpt(targetChip, "sm_70");
            setOpt(targetFeatures, "+ptx70");
            break;
          default:
            break;
          }
        }));

    static opt<bool, true> emitLLVM(cat(cmdCat), "offload-llvm",
                                    desc("Emit LLVM IR:"),
                                    location(options.emitLLVM), init(true));
  }

  ParOptionsImpl options;
  llvm::cl::OptionCategory *cmdCat{};
};

llvm::ManagedStatic<ParOptionsCL> clInvocation;
} // namespace

ParOptions::ParOptions() : impl(&(clInvocation->options)) {}

ParBackend ParOptions::getBackend() const {
  return static_cast<ParBackend>(impl->parBackend);
}

StringRef ParOptions::getTriple() const { return impl->targetTriple; }

StringRef ParOptions::getChip() const { return impl->targetChip; }

StringRef ParOptions::getTargetFeatures() const { return impl->targetFeatures; }

int ParOptions::getOptLevel() const { return impl->optimizationLevel; }

StringRef ParOptions::getToolkitPath() const { return impl->toolkitPath; }

StringRef ParOptions::getToolOpts() const { return impl->toolOpts; }

llvm::ArrayRef<std::string> ParOptions::getLinkLibs() const {
  return impl->linkLibs;
}

bool ParOptions::isSequential() const { return getBackend() == seq; }

bool ParOptions::isHost() const { return getBackend() == host; }

bool ParOptions::isOffload() const { return isNVPTX() || isAMDGPU(); }

bool ParOptions::isNVPTX() const { return getBackend() == nvptx; }

bool ParOptions::isAMDGPU() const { return getBackend() == amdgpu; }

bool ParOptions::isSPIRV() const { return getBackend() == spirv; }

bool ParOptions::emitLLVM() const { return impl->emitLLVM; }

llvm::cl::OptionCategory *xblang::par::registerParOptions() {
  return clInvocation->cmdCat;
}
