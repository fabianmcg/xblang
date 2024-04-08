#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

using namespace llvm;

#include "xblang/Driver/Driver.h"

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;
  int status = xblang::Driver::run(llvm::ArrayRef<const char *>(argv, argc));
#ifndef NDEBUG
  if (status)
    llvm::errs() << "return code: " << status << "\n";
#endif
  return status;
}
