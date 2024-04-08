#ifndef XBLANG_LANG_PAR_FRONTEND_OPTIONS_H
#define XBLANG_LANG_PAR_FRONTEND_OPTIONS_H

#include "xblang/Support/LLVM.h"

namespace llvm {
namespace cl {
class OptionCategory;
}
} // namespace llvm

namespace xblang {
namespace par {
struct ParOptionsImpl;

class ParOptions {
public:
  /// Parallelization back-end.
  typedef enum : uint8_t {
    seq,
    host,
    mp = host,
    nvptx,
    amdgpu,
    spirv,
  } ParBackend;

  /// Constructs the object with the options specified in the command line.
  ParOptions();

  /// Returns the parallelization back-end.
  ParBackend getBackend() const;

  /// Return the target triple.
  StringRef getTriple() const;

  /// Return the target chip.
  StringRef getChip() const;

  /// Returns the target features.
  StringRef getTargetFeatures() const;

  /// Returns the optimization level.
  int getOptLevel() const;

  /// Returns the toolkit path.
  StringRef getToolkitPath() const;

  /// Returns the tool options.
  StringRef getToolOpts() const;

  /// Returns the bitcode libraries to link to.
  llvm::ArrayRef<std::string> getLinkLibs() const;

  /// Returns whether the parallelization back-end is sequential.
  bool isSequential() const;

  /// Returns whether the parallelization back-end is a host.
  bool isHost() const;

  /// Returns whether the parallelization back-end is offload.
  bool isOffload() const;

  /// Returns whether the offload back-end is NVPTX.
  bool isNVPTX() const;

  /// Returns whether the offload back-end is AMDGPU.
  bool isAMDGPU() const;

  /// Returns whether the offload back-end is SPIR-V.
  bool isSPIRV() const;

  /// Returns whether to emit LLVM IR instead of a binary.
  bool emitLLVM() const;

private:
  const ParOptionsImpl *impl;
};

/// Registers the front-end options.
llvm::cl::OptionCategory *registerParOptions();
} // namespace par
} // namespace xblang
#endif /* XBLANG_LANG_PAR_FRONTEND_OPTIONS_H */
