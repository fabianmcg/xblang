#ifndef XBLANG_DRIVER_DRIVER_H
#define XBLANG_DRIVER_DRIVER_H

#include "xblang/Frontend/CompilerInstance.h"
#include "xblang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/ArrayRef.h"

namespace xblang {
class Driver {
public:
  Driver() = default;
  ~Driver() = default;
  static int run(llvm::ArrayRef<const char *> cmdArgs);

private:
  bool parseArgs();

  llvm::ArrayRef<const char *> cmdArgs;
  CompilerInvocation invocation;
  std::unique_ptr<CompilerInstance> compiler;
};
} // namespace xblang

#endif
