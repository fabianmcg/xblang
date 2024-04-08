//===- Passes.td - Syntax dialect passes -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYNTAX_PASSES
#define SYNTAX_PASSES

include "mlir/Pass/PassBase.td"

def ProcessSyntax : Pass<"process-syntax"> {
  let summary = "Canonicalizes a syntax.";
  let description = [{
    This pass processes the syntax by inlining macros, eliminating common
    sub-expressions and performing canonicalization.
    It also transforms syntax rules to lexing rules.
  }];
  let options = [
    Option<"rulesToLex", "rules-to-lex", "bool", /*default=*/"false",
           "Transforms syntax rules to lexing rules">
  ];
}

def LexToDFA : Pass<"lex-to-dfa", "DFAOp"> {
  let summary = "Transforms a regular grammar into a DFA.";
  let description = [{
    This pass transforms a regular grammar into a DFA.
  }];
}

def MinimizeDFA : Pass<"minimize-dfa", "DFAOp"> {
  let summary = "Minimizes the DFA.";
}

def AnalyzeSyntax : Pass<"analyze-syntax", "ParserOp"> {
  let summary = "Analyzes a syntax and adds metadata.";
}

#endif // SYNTAX_PASSES
