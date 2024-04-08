#ifndef UNITTESTS_BASIC_SOURCEMANAGER_CPP
#define UNITTESTS_BASIC_SOURCEMANAGER_CPP

#include "xblang/Basic/SourceManager.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>

namespace {
TEST(SrcManager, report) {
  std::string error;
  llvm::raw_string_ostream os(error);
  mlir::MLIRContext context;
  xblang::SourceManager manager(&context, &os);
  manager.registerDiagnosticsHandler();
  xblang::Source *source =
      manager.createSource("Lorem\n\nipsum\nipsum", "buffer");
  EXPECT_TRUE(!!source);
  auto loc =
      mlir::OpaqueLoc::get(source->getState().getLoc().loc.getPointer() + 7,
                           mlir::UnknownLoc::get(&context));
  mlir::emitError(loc) << "my error";
  EXPECT_TRUE(error.find("buffer:3:1: error: my error\nipsum\n^") == 0);
}
} // namespace

#endif
