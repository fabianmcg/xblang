#include "xblang/Driver/Driver.h"

#include "mlir/IR/AsmState.h"
#include "xblang/Lang/Parallel/Frontend/Options.h"
#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <vector>

using namespace xblang;

namespace {
template <class T, class Alloc, class U>
constexpr typename std::vector<T, Alloc>::size_type
erase(std::vector<T, Alloc> &c, const U &value) {
  auto it = std::remove(c.begin(), c.end(), value);
  auto r = std::distance(it, c.end());
  c.erase(it, c.end());
  return r;
}
} // namespace

bool Driver::parseArgs() {
  std::vector<llvm::cl::OptionCategory *> categories;
  categories.push_back(CompilerInvocation::registerCLOpts());
  categories.push_back(par::registerParOptions());
  mlir::registerAsmPrinterCLOptions();
  ::erase(categories, static_cast<llvm::cl::OptionCategory *>(nullptr));
  llvm::cl::HideUnrelatedOptions(categories);
  auto status =
      llvm::cl::ParseCommandLineOptions(cmdArgs.size(), cmdArgs.data());
  if (status)
    invocation = CompilerInvocation::getCLOpts();
  return status;
}

int Driver::run(llvm::ArrayRef<const char *> cmdArgs) {
  Driver driver;
  driver.cmdArgs = cmdArgs;
  auto status = driver.parseArgs();
  if (status) {
    driver.compiler = std::unique_ptr<CompilerInstance>(
        new CompilerInstance(driver.invocation));
    return driver.compiler->run();
  } else
    return 1;
}
