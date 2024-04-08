//===- XLGBase.td - Tablegen XLG base classes --------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tablegen XLG base classes.
//
//===----------------------------------------------------------------------===//

#ifndef XBLANG_XLG_XLGBASE_TD
#define XBLANG_XLG_XLGBASE_TD

include "mlir/IR/OpBase.td"
include "xblang/Interfaces/Concept.td"

// Generic XLG concept type.
def XLG_ConceptType :
  Type<
    CPred<"::mlir::isa<::xblang::xlg::ConceptType>($_self)">,
    "xlg.concept",
    "::xblang::xlg::ConceptType"
  >;

// XLG concept type.
class XLGConceptClass<string className, string summary = "xlg.concept"> :
    Type<CPred<"::xblang::xlg::isClass<"# className #">($_self)">,
         summary, "::xblang::xlg::ConceptType">;

// XLG concept trait.
class ConceptTrait<string concept> : NativeOpTrait<"ConceptTrait"> {
  // The C++ namespace to use for this concept.
  let cppNamespace = "::xblang::xlg";
  // Additional code that will be added to the public part of the generated
  // C++ code of the class.
  let extraConcreteClassDeclaration = "using ConceptType = " # concept # ";" # [{
    static ::xblang::Concept *getConcept(::xblang::XBContext *ctx) {
      return ConceptType::get(ctx);
    }
  }];
  let extraConcreteClassDefinition = [{}];
}

// XLG base concept class.
class ConceptBase<string name> : Type<CPred<"">, "xlg.class", "::xblang::xlg::ConceptType"> {
  // Class name of the concept.
  string className = name;
  // The C++ namespace to use for this concept.
  string cppNamespace = "::xblang::xlg";
  // Type predicate.
  let predicate = CPred<"::xblang::xlg::isClass<"# cppNamespace # "::"
                                                 # className #">($_self)">;
  // MLIR native trait for the concept.
  ConceptTrait conceptTrait = ConceptTrait<cppNamespace # "::" # className>;
}

// Language concept.
class Concept<string name, string mnemonic, list<Concept> conceptList = []> :
    ConceptBase<name> {
  // Short summary describing the concept.
  string summary = "";

  // AOptional longer description of the concept.
  string description = "";

  // The mnemonic of the concept.
  string conceptName = mnemonic;

  // The mnemonic of the dialect.
  string dialectName = ?;

  // List of parent concepts.
  list<Concept> parentConcepts = conceptList;

  // Traits local to the concept.
  list<Trait> traits = [];

  // Dag containing the arguments of the concept.
  dag args = (ins);

  // Additional code that will be added to the public part of the generated
  // C++ code of the class declaration.
  code extraClassDeclaration = ?;

  // Additional code that will be added to the generated source file. The
  // generated code is placed inside the op's C++ namespace. `$cppClass` is
  // replaced by the concepts's C++ class name.
  code extraClassDefinition = ?;

  // Determines if the concept also names a pure construct.
  bit pureConstruct = 0;
}

// Utility class for obtaining all the traits of a concept.
class ConceptTraitFolder<Concept concept> {
  list<Trait> emptyList = [];
  list<Trait> traits = !listconcat(
    concept.traits,
    !foldl(
      /*init=*/emptyList,
      /*list=*/!foreach(
        /*var=*/c,
        /*sequence=*/concept.parentConcepts,
        /*expr=*/ConceptTraitFolder<c>.traits
      ),
      /*acc=*/traitList,
      /*var=*/t,
      /*expr=*/!listconcat(traitList, t)
    )
  );
}

// Utility class for obtaining all the traits of a list of concepts.
class ConceptTraitListFolder<list<Concept> concepts> {
  list<Trait> emptyList = [];
  list<Trait> traits = !foldl(
    /*init=*/emptyList,
    /*list=*/concepts,
    /*acc=*/traitList,
    /*var=*/concept,
    /*expr=*/!listconcat(traitList, ConceptTraitFolder<concept>.traits)
  );
}

// Utility class for obtaining all the args of a concept.
class ConceptArgFolder<Concept concept> {
  dag emptyDag = (ins);
  dag args = !con(
    !foldl(
      /*init=*/emptyDag,
      /*list=*/!foreach(
        /*var=*/c,
        /*sequence=*/concept.parentConcepts,
        /*expr=*/ConceptArgFolder<c>.args
      ),
      /*acc=*/dagArgs,
      /*var=*/d,
      /*expr=*/!con(dagArgs, d)
    ),
    concept.args
  );
}

// Utility class for obtaining all the args of a list of concepts.
class ConceptArgListFolder<list<Concept> concepts> {
  dag emptyDag = (ins);
  dag args = !foldl(
    /*init=*/emptyDag,
    /*list=*/concepts,
    /*acc=*/dagArgs,
    /*var=*/concept,
    /*expr=*/!con(dagArgs, ConceptArgFolder<concept>.args)
  );
}

// Language dialect concept.
class DialectConcept<Dialect dialect,
                     string name,
                     string mnemonic,
                     list<Concept> conceptList = []> :
    Concept<name, mnemonic, conceptList> {
  Dialect dialectConcept = dialect;
  let cppNamespace = dialectConcept.cppNamespace;
  let dialectName = dialectConcept.name;
}

// Language construct.
class Construct<Dialect dialect,
                string name,
                string mnemonic,
                list<Concept> conceptList = [],
                list<Trait> props = []> :
    Op<dialect, mnemonic, []>,
    DialectConcept<dialect, name, mnemonic, conceptList> {
  // Args inherited from the concepts.
  dag constructArgs = ConceptArgListFolder<conceptList>.args;

  // Construct specific args.
  dag args = (ins);

  // MLIR Op arguments.
  let arguments = !con(constructArgs, args);

  // MLIR Op results.
  let results = (outs XLG_ConceptType:$conceptClass);

  // Set the construct's traits.
  let traits = !listconcat(props,
                           ConceptTraitListFolder<conceptList>.traits,
                           [conceptTrait, ConceptInterface]);
}

#endif // XBLANG_XLG_XLGBASE_TD
