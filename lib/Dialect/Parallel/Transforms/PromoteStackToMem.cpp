#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"

namespace mlir {
namespace par {
#define GEN_PASS_DEF_PROMOTESTACKTOMEM
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"

namespace {
struct VarToMem : public OpRewritePattern<xblang::xb::VarOp> {
  using Base = OpRewritePattern<xblang::xb::VarOp>;
  using Op = xblang::xb::VarOp;
  using Base::Base;
  LogicalResult match(Op op) const final;
  void rewrite(Op op, PatternRewriter &rewriter) const final;
};

class PromoteStackToMem
    : public impl::PromoteStackToMemBase<PromoteStackToMem> {
public:
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

void PromoteStackToMem::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<VarToMem>(&getContext());
  FrozenRewritePatternSet patternSet(std::move(patterns));
  auto result = mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet);
  if (result.failed())
    signalPassFailure();
}

LogicalResult VarToMem::match(Op op) const {
  if (!op->getParentOp()->hasTrait<FunctionOpInterface::Trait>())
    return failure();
  if (isa<TensorType>(op.getType()))
    return failure();
  if (op->getAttr("external_stack"))
    return failure();
  for (auto user : op->getUsers())
    if (isa<par::MapOp>(user))
      return success();
  return failure();
}

void VarToMem::rewrite(Op op, PatternRewriter &rewriter) const {
  auto type = op.getType();
  auto ptrType = xblang::xb::PointerType::get(op.getType());
  auto refType = isa<xblang::xb::ReferenceType>(op.getType())
                     ? op.getType()
                     : xblang::xb::ReferenceType::get(op.getType());
  auto addressType = xblang::xb::AddressType::get(op.getContext());
  auto indexType = rewriter.getIndexType();

  auto name = op.getSymName().str() + "_stub";
  auto var = rewriter.create<xblang::xb::VarOp>(
      op.getLoc(), refType, name, type, op.getKind(), op.getInit());
  Value sz =
      rewriter.create<xblang::xb::SizeOfOp>(op.getLoc(), indexType, type);
  Value c1 = rewriter.create<xblang::xb::ConstantOp>(op.getLoc(),
                                                     rewriter.getIndexAttr(1));
  Value varAddress = rewriter.create<xblang::xb::UnaryOp>(
      op.getLoc(), ptrType, xblang::UnaryOperator::Address, var.getResult());
  varAddress =
      rewriter.create<xblang::xb::CastOp>(op.getLoc(), addressType, varAddress);
  Value address = rewriter
                      .create<xblang::xb::CallOp>(
                          op.getLoc(), addressType, "__xblangAlloca",
                          ValueRange({varAddress, sz, c1}))
                      .getResults()[0];
  Value stackAddr = address;
  address = rewriter.create<xblang::xb::CastOp>(op.getLoc(), ptrType, address,
                                                false, true);
  auto uop = rewriter.create<xblang::xb::UnaryOp>(
      op.getLoc(), refType, xblang::UnaryOperator::Dereference, address);

  auto varRef = rewriter.replaceOpWithNewOp<xblang::xb::VarOp>(
      op, refType, op.getSymName(), refType, op.getKind(), uop);

  varRef->setAttr("external_stack", rewriter.getUnitAttr());

  {
    rewriter.setInsertionPoint(&uop->getParentRegion()->back().back());
    rewriter.create<xblang::xb::CallOp>(uop.getLoc(), TypeRange(),
                                        "__xblangDealloca",
                                        ValueRange({varAddress, stackAddr}));
  }
}
} // namespace par
} // namespace mlir
