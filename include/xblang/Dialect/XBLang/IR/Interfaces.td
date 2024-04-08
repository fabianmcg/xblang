#ifndef XBLANG_DIALECT_XBLANG_IR_INTERFACES_TD
#define XBLANG_DIALECT_XBLANG_IR_INTERFACES_TD

include "mlir/IR/OpBase.td"

def ImplicitCastInterface : OpInterface<"ImplicitCast"> {
  let description = [{
  }];
  let cppNamespace = "::xblang::xb";
  let methods = [
    InterfaceMethod<
    /* Description */ [{}],
    /* Return Type */ "::mlir::Type",
    /* Name */ "getImplicitCast",
    /* Arguments */ (ins "unsigned":$arg),
    /* Method Body */ [{}],
    /* Default Implementation */ [{
      auto op = $_op;
      assert(arg < op->getNumOperands());
      auto type = op->getOperand(arg).getType();
      if (auto refType = dyn_cast<::xblang::xb::ReferenceType>(type))
        return refType.getPointee();
      return nullptr;
    }]
    >
  ];
}

def ImplicitCastTrait: DeclareOpInterfaceMethods<ImplicitCastInterface, ["getImplicitCast"]>;

#endif
