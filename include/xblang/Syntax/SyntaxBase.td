#ifndef XBLANG_SYNTAX_SYNTAXBASE_TD
#define XBLANG_SYNTAX_SYNTAXBASE_TD

include "mlir/IR/Utils.td"
include "mlir/IR/Traits.td"

//===----------------------------------------------------------------------===//
// Lexer definitions
//===----------------------------------------------------------------------===//

// Lexer token
class Token<string id, list<string> a = []> {
  // Identifier of the token
  string identifier = id;
  // Token aliases
  list<string> aliases = a;
}

// Lexer token class
class TokenClass<string id, list<Token> toks = []> {
  // Identifier of the token class
  string identifier = id;
  // List of tokens in the class
  list<Token> tokens = toks;
}

// Lexer definition
class Definition<string id = "", string expr = ""> {
  // Identifier of the definition
  string identifier = id;
  // Rule used for lexing the definition
  string expression = expr;
}

// Lexer rule
class Rule<string id, string r = "", code a = "", list<string> aliases = []> :
    Token<id, aliases> {
  // Rule used for lexing
  string rule = r;
  // Default action to be performed upon lexing the rule
  code defaultAction = [{
    spelling = getSpelling(beginLoc, state.getLoc());
    return $token;
  }];
  // Action invoked by the lexer after lexing the rule
  string action = !if(!empty(a), defaultAction, a);
}

// Class for specifying keywords. Keywords only define tokens.
class Keyword<string id, string spelling> :
  Token<id, [spelling]> {
}

// Helper class for defining lexing literals.
class Literal<string id,
              string literal,
              code action = "",
              list<string> aliases = []> :
  Rule<id, "'" # literal # "'", action, [literal] # aliases> {
}

// Finite automata
class FiniteAutomata<string id> {
  // Identifier of the automata
  string name = id;
  // Determines whether the automata ignores white spaces
  bit ignoreWhitespace = 1;
  // Determines whether the automata is created inside a while loop (recommended)
  bit loop = 1;
  // List of definitions used in the automata
  list<Definition> definitions;
  // List of rules the automata while lex
  list<Rule> rules;
  // The action to perform upon encountering an error
  code errorAction = [{ return Invalid; }];
}

// Class for defining lexers
class Lexer<string id> {
  // Identifier of the lexer
  string name = id;
  // The C++ namespace where the lexer belongs
  string cppNamespace = ?;
  // List of tokens used by the lexer
  list<Token> tokens;
  // List of tokens classes used by the lexer
  list<TokenClass> tokenClasses = [];
  // List of automatons accepted by the lexer
  list<FiniteAutomata> automatons;
  // Extra class declarations in the lexer
  code extraClassDeclaration = ?;
  // Extra class definitions in the lexer
  code extraClassDefinition = ?;
  // Determines whether to implement the lexer
  bit implement = 1;
}

//===----------------------------------------------------------------------===//
// Parser definitions
//===----------------------------------------------------------------------===//

// Parser traits.
class ParserTrait<string name, string namespace> : NativeTrait<name, ""> {
  string cppNamespace = namespace;
} 

// Parser production
class Production<string n, string rt, bit i = 1> {
  // Identifier of the production
  string identifier = n;
  // The return type of the production
  string returnType = rt;
  // Whether the generator should implement the production
  bit implement = i;
  // The parsing rule
  code rule = ?;
  // The arguments of the production.
  dag arguments = (ins);
}

// Parser macro
class ParserMacro<string n, list<string> a = [], code e = ""> {
  // Identifier of the macro
  string identifier = n;
  // Arguments of the macro
  list<string> arguments = a;
  // The syntax expression
  code expression = e;
}

class Parser<string n, Lexer l, list<ParserTrait> t = []> {
  // Identifier of the lexer
  string name = n;
  // The C++ namespace where the lexer belongs
  string cppNamespace = ?;
  // Start symbol of the parser
  string startSymbol = ?;
  // Default lexer used by the parser
  Lexer lexer = l;
  // Extra class declarations in the lexer
  code extraClassDeclaration = ?;
  // Extra class definitions in the lexer
  code extraClassDefinition = ?;
  // Determines whether to implement the lexer
  bit implement = 1;
  // List of traits the parser implements
  list<ParserTrait> traits = t;
  // List of productions used by the parser
  list<Production> productions;
  // List of productions used by the parser
  list<ParserMacro> macros;
  // Default lexing token to match when building the parser
  string defaultToken = "";
}

#endif
