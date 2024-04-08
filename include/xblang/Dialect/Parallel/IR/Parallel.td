#ifndef XBLANG_DIALECT_PARALLEL_IR_PARALLEL_TD
#define XBLANG_DIALECT_PARALLEL_IR_PARALLEL_TD

include "xblang/Dialect/Parallel/IR/Dialect.td"
include "xblang/Dialect/Parallel/IR/Enums.td"
include "xblang/Dialect/XBLang/IR/Type.td"

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Par Attrs
//===----------------------------------------------------------------------===//
def MapKindArrayAttr :
  TypedArrayAttrBase<MapKindEnum, "map kind array attribute"> {
  let constBuilderCall = ?;
}

def Par_Offload : Par_Attr<"Offload", "offload", [
    OffloadingTranslationAttrTrait,
    DeclareAttrInterfaceMethods<OffloadingLLVMTranslationAttrInterface>
  ]> {
  let parameters = (ins StringRefParameter<"Offload kind.", "\"bin\"">:$kind);
  let assemblyFormat = [{ (`<` $kind^ `>`)? }];
  let genVerifyDecl = 0;
}

//===----------------------------------------------------------------------===//
// Par atomic Op
//===----------------------------------------------------------------------===//

def Par_AtomicOp: Par_Op<"atomic", []> {
  let summary = "Atomic operation";
  let description = [{}];
  let arguments = (ins
    AnyType:$destination,
    AnyType:$value,
    AtomicOpsEnum:$op
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `(` $op `,`
        $destination `:` type($destination) `,`
        $value type($value) `)`
      `:` type($result) attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Par data region Op
//===----------------------------------------------------------------------===//

def Par_DataRegionOp : Par_Op<"data_region", [
    AttrSizedOperandSegments,
    AutomaticAllocationScope,
    SingleBlock,
    AffineScope]> {
  let arguments = (ins Variadic<AnyType>:$variables,
                       Variadic<AnyType>:$memMappings,
                       OptionalAttr<MapKindArrayAttr>:$mappings,
                       Variadic<AddressType>:$queues);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    `map` `(`custom<MapList>($variables, type($variables), $memMappings, type($memMappings), $mappings, $queues, type($queues))`)`
    attr-dict-with-keyword
    custom<RegionWithImplicitYield>($body)
  }];
  let builders = [
    OpBuilder<(ins "ValueRange":$variables, "ValueRange":$memMappings, "ArrayAttr":$mappings), [{
      auto queue = $_builder.create<DefaultQueueOp>($_builder.getUnknownLoc(), xblang::xb::AddressType::get($_builder.getContext())); 
      SmallVector<Value> queues(variables.size(), queue);
      build($_builder, $_state, variables, memMappings, mappings, queues);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Par default queue Op
//===----------------------------------------------------------------------===//

def Par_DefaultQueueOp: Par_Op<"default_queue", [Pure, ConstantLike]> {
  let summary = "Queue operation";
  let description = [{}];
  let results = (outs AddressType:$result);
  let assemblyFormat = [{
  type($result) attr-dict
  }];
  let hasVerifier = 0;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Par dim Op
//===----------------------------------------------------------------------===//

def Par_DimOp: Par_Op<"dim", []> {
  let summary = "Dim operation";
  let description = [{}];
  let arguments = (ins
    ParallelHierarchyEnum:$rank,
    I8Attr:$entry
  );
  let results = (outs SI32:$result);
  let assemblyFormat = [{
    $rank $entry attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Par func Op
//===----------------------------------------------------------------------===//

def Par_FuncOp: Par_Op<"func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove, Symbol]> {
  let summary = "";
  let description = [{ }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs
                       );
  let regions = (region AnyRegion:$body);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }
    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    LogicalResult verifyType();
    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Par id Op
//===----------------------------------------------------------------------===//

def Par_IdOp: Par_Op<"id", []> {
  let summary = "Id operation";
  let description = [{}];
  let arguments = (ins
    ParallelHierarchyEnum:$rank,
    I8Attr:$entry
  );
  let results = (outs SI32:$result);
  let assemblyFormat = [{
    $rank $entry attr-dict
  }];
  let builders = [
    OpBuilder<(ins "::mlir::par::ParallelHierarchy":$kind,
                   CArg<"int", "0">:$entry), [{
      build($_builder, $_state, $_builder.getIntegerType(32, true),
            kind, entry);
    }]>
  ];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Par loop Op
//===----------------------------------------------------------------------===//

def Par_LoopInfo : Property<"SmallVector<LoopInfo>", "Loop information"> {
  let interfaceType = "::llvm::ArrayRef<LoopInfo>";
  let assignToStorage = [{
  	$_storage.assign(std::begin($_value), std::end($_value))
  }];
  let hashProperty = [{
  	llvm::hash_combine_range(std::begin($_storage), std::end($_storage))
  }];
  let writeToMlirBytecode = writeMlirBytecodeWithConvertToAttribute;
  let readFromMlirBytecode = readMlirBytecodeUsingConvertFromAttribute;
}

def Par_LoopOp : Par_Op<"loop", [
      AutomaticAllocationScope,
      RecursiveMemoryEffects,
      DeclareOpInterfaceMethods<LoopLikeOpInterface>,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
      DeclareOpInterfaceMethods<RegionBranchOpInterface>,
      RecursivelySpeculatable,
      SameVariadicOperandSize
    ]> {
  let summary = "Par Loop Op";
  let description = [{}];
  let arguments = (ins Variadic<AnyType>:$begin,
                       Variadic<AnyType>:$end,
                       Variadic<AnyType>:$step,
                       Par_LoopInfo:$loopInfo,
                       DefaultValuedAttr<ParallelHierarchyEnum,
                       					 "ParallelHierarchy::automatic"
                       					>:$parallelization);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    (`<` $parallelization^ `>`)?
    attr-dict
    custom<Loop>($begin, type($begin),
                 $end, type($end),
                 $step, type($step),
                 $loopInfo,
                 $body)
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ArrayRef<LoopBuilder>":$loops, 
    			   CArg<"ParallelHierarchy",
    			        "ParallelHierarchy::automatic">:$parallelization)>
  ];
  let extraClassDeclaration = [{
    LoopDescriptor getLoop(unsigned indx = 0);
    SmallVector<LoopDescriptor> getLoops();
  }];
}

//===----------------------------------------------------------------------===//
// Par make queue Op
//===----------------------------------------------------------------------===//

def Par_MakeQueueOp: Par_Op<"make_queue", []> {
  let summary = "Queue operation";
  let description = [{}];
  let results = (outs AddressType:$result);
  let assemblyFormat = [{
  type($result) attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Par map Op
//===----------------------------------------------------------------------===//

def Par_MapOp: Par_Op<"map", []> {
  let summary = "Map operation";
  let description = [{}];
  let arguments = (ins
    AnyType:$memReference,
    MapKindEnum:$kind,
    Optional<AddressType>:$queue
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `(` $kind `,`  $memReference `:` type($memReference) (`,` $queue^ type($queue))? `)` 
    `:` type($result) attr-dict
  }];
  let hasVerifier = 0;
  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$memReference, "MapKind":$kind), [{
      build($_builder, $_state, type, memReference, kind, nullptr);
    }]>,
    OpBuilder<(ins "Type":$type, "Value":$memReference, "MapKindAttr":$kind), [{
      build($_builder, $_state, type, memReference, kind, nullptr);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Par reduce Op
//===----------------------------------------------------------------------===//

def Par_ReduceOp: Par_Op<"reduce", []> {
  let summary = "Reduce operation";
  let description = [{}];
  let arguments = (ins
    AnyType:$value,
    Optional<AnyType>:$init,
    ReduceOpsEnum:$op,
    OptionalAttr<ParallelHierarchyEnum>:$rank
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `(` $op (`,` $rank^)? `,`
        $value type($value) (`,` $init ^ type($init))? `)`
      `:` type($result) attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Par region Op
//===----------------------------------------------------------------------===//

def Par_RegionOp : Par_Op<"region", [
      AffineScope,
      AttrSizedOperandSegments,
      AutomaticAllocationScope,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
      SingleBlock
    ]> {
  let arguments = (ins Variadic<Index>:$tensorDim,
                       Variadic<Index>:$matrixDim,
                       DefaultValuedAttr<DataSharingKindEnum,
                        "DataSharingKind::Private">:$defaultDataSharing,
                       Variadic<ReferenceType>:$firstPrivateVars,
                       OptionalAttr<DictArrayAttr>:$firstPrivateVarsAttrs,
                       Variadic<ReferenceType>:$privateVars,
                       OptionalAttr<DictArrayAttr>:$privateVarsAttrs,
                       Variadic<ReferenceType>:$sharedVars,
                       OptionalAttr<DictArrayAttr>:$sharedVarsAttrs,
                       Variadic<ReferenceType>:$mappedVars,
                       Variadic<PointerType>:$varMappings,
                       Optional<AddressType>:$queue,
                       IntProperty<"int64_t">:$numMatrixVars,
                       StrArrayAttr:$attributionsNames);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    attr-dict custom<Queue>($queue, type($queue))
    (`tensor``(` $tensorDim^ `)`)?
    (`matrix``(` $matrixDim^ `)`)?
    (`default` `(` $defaultDataSharing^ `)`)?
    oilist(
     `firstprivate` custom<ParRegionArgs>($firstPrivateVars,
                                          type($firstPrivateVars),
                                          $firstPrivateVarsAttrs) |
     `private` custom<ParRegionArgs>($privateVars,
                                     type($privateVars),
                                     $privateVarsAttrs) |
     `shared` custom<ParRegionArgs>($sharedVars,
                                    type($sharedVars),
                                    $sharedVarsAttrs) |
     `mappings` custom<RegionMappings>($mappedVars,
                                       type($mappedVars),
                                       $varMappings,
                                       type($varMappings))
    )
    custom<ParRegionBody>($body, $numMatrixVars, $attributionsNames)
  }];
  let builders = [
    OpBuilder<(ins
      "ValueRange":$tensorDim,
      "ValueRange":$matrixDim,
      CArg<"Value", "{}">:$queue,
      CArg<"ValueRange", "ValueRange()">:$firstprivateVars,
      CArg<"ValueRange", "ValueRange()">:$privateVars,
      CArg<"ValueRange", "ValueRange()">:$sharedVars,
      CArg<"bool", "true">:$addBlock,
      CArg<"DataSharingKind", "DataSharingKind::Private">:$defaultDataSharing,
      CArg<"std::pair<TypeRange, ArrayRef<Location>>", "{}">:$matrixVars,
      CArg<"std::pair<TypeRange, ArrayRef<Location>>", "{}">:$tensorVars,
      CArg<"ArrayRef<StringRef>", "{}">:$varNames,
      CArg<"ValueRange", "ValueRange()">:$mappedVars,
      CArg<"ValueRange", "ValueRange()">:$varMappings,
      CArg<"ArrayAttr", "{}">:$firstprivateVarsAttrs,
      CArg<"ArrayAttr", "{}">:$privateVarsAttrs,
      CArg<"ArrayAttr", "{}">:$sharedVarsAttrs
    )>
  ];
  let extraClassDeclaration = [{
    /// Returns the number of variables with a matrix scope.
    int64_t getNumMatrixVariables();

    /// Returns the number of variables with a tensor scope.
    int64_t getNumTensorVariables();

    /// Returns the matrix variables.
    ArrayRef<BlockArgument> getMatrixVariables();

    /// Returns the tensor variables.
    ArrayRef<BlockArgument> getTensorVariables();

    /// Add a matrix variable.
    BlockArgument addMatrixVariable(Type type, Location loc);

    /// Add a tensor variable.
    BlockArgument addTensorVariable(Type type, Location loc);

    /// Add a firstprivate variable.
    void addFirstprivateVariable(ValueRange variables);

    /// Add a private variable.
    void addPrivateVariable(ValueRange variables);

    /// Add a shared variable.
    void addSharedVariable(ValueRange variables);

    /// Add mapped variables.
    void addMappings(ValueRange variables, ValueRange mappings);
  }];
}


//===----------------------------------------------------------------------===//
// Par sync Op
//===----------------------------------------------------------------------===//

def Par_SyncOp: Par_Op<"sync", []> {
  let summary = "Sync operation";
  let description = [{}];
  let arguments = (ins
    ParallelHierarchyEnum:$rank
  );
  let assemblyFormat = [{
    $rank attr-dict
  }];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Par wait Op
//===----------------------------------------------------------------------===//

def Par_WaitOp: Par_Op<"wait", []> {
  let summary = "Wait operation";
  let description = [{}];
  let arguments = (ins Variadic<AddressType>:$tokens, UnitAttr:$destroy);
  let assemblyFormat = [{
   (`destroy` $destroy^)?  `(` $tokens type($tokens) `)` attr-dict
  }];
  let hasVerifier = 0;
}

#endif
