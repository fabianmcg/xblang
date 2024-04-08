#include "xblang/ADT/Identifier.h"
#include "llvm/Support/raw_ostream.h"

namespace xblang {
llvm::raw_ostream &operator<<(llvm::raw_ostream &ost,
                              const Identifier &identifier) {
  ost << identifier.toString();
  return ost;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &ost,
                              const QualifiedIdentifier &identifier) {
  for (auto iv : llvm::enumerate(identifier)) {
    ost << iv.value();
    if (iv.index() + 1 < identifier.size())
      ost << "::";
  }
  return ost;
}
} // namespace xblang
