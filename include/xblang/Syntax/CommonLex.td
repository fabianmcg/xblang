#ifndef XBLANG_COMMON_LEX
#define XBLANG_COMMON_LEX

include "xblang/Syntax/SyntaxBase.td"

//===----------------------------------------------------------------------===//
// Common definitions
//===----------------------------------------------------------------------===//

// Decimal digit
def DigitStx : Definition<"digit", "[0-9]">;

// Binary digit
def BinaryDigitStx : Definition<"binary_digit", "[0-1]">;

// Octal digit
def OctalDigitStx : Definition<"octal_digit", "[0-8]">;

// Hex digit
def HexDigitStx : Definition<"hex_digit", "[0-9a-fA-F]">;

// Non-zero decimal digit
def NonZeroStx : Definition<"nonzero", "[1-9]">;

// Alphabetic character
def AlphaStx : Definition<"alpha", "[a-zA-Z]">;

// Alpha-numeric character
def AlphaNumStx : Definition<"alphanum", "[a-zA-Z0-9]">;

// Digit sequence without leading zeros
def DigitSeqStx : Definition<"digit_seq", "(nonzero digit*) | '0'">;

// Digit sequence allowing leading zeros
def DigitSeqZStx : Definition<"digit_seq_z", "digit* | '0'">;

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

// Class for generating int literals
class IntLiteral<string sfx = ""> {
  // Suffix appended to the literal
  string suffix = sfx;
  string lexSuffix = !if(!empty(suffix), "", "'" # suffix  # "'");
  // Decimal literal
  Rule decimalLiteral = Rule<
    "IntLiteral" # suffix,
    [{ (nonzero digit* | '0') }] # lexSuffix
  >;
  // Binary literal
  Rule binaryLiteral = Rule<
    "BinaryIntLiteral" # suffix,
    [{ ('0b' | '0B') binary_digit+ }] # lexSuffix
  >;
  // Octal literal
  Rule octalLiteral = Rule<
    "OctalIntLiteral" # suffix,
    [{ ('0o' | '0O') octal_digit+ }] # lexSuffix
  >;
  // Hex literal
  Rule hexLiteral = Rule<
    "HexIntLiteral" # suffix,
    [{ ('0x' | '0X') hex_digit+ }] # lexSuffix
  >;
  // Helper field with all the literals
  list<Rule> rules = [decimalLiteral, binaryLiteral, octalLiteral, hexLiteral];
}

// Class for generating float literals
class FloatLiteral<string sfx = ""> {
  // Suffix appended to the literal
  string suffix = sfx;
  string lexSuffix = !if(!empty(suffix), "", "'" # suffix  # "'");
  // Floating-point literal
  Rule floatLiteral = Rule<
    "FloatLiteral" # suffix,
    [{ (
      (digit_seq [eE] [+\-]? digit_seq_z) |
      (digit_seq '.' ([eE] [+\-]? digit_seq_z)?) |
      (digit_seq? '.' digit+ ([eE] [+\-]? digit_seq_z)?)
    ) }] # lexSuffix
  >;
  // Helper field with all the literals
  list<Rule> rules = [floatLiteral];
}

// String literal
def StringLiteralStx : Rule<"StringLiteral"> {
  let rule = [{ ('"' | '\'') }];
  let action = [{
    if (Invalid == parseString(state, spelling))
      return Invalid;
    return $token;
  }];
}

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

// C-style comment
def CommentStx : Rule<"Comment"> {
  let rule = [{ '/*' }];
  let action = [{
    bool terminated = false;
    while (*state) {
      if (state[0] == '*' && state.at(1) == '/') {
        state.advance();
        state.advance();
        terminated = true;
        break;
      }
      state.advance();
    }
    if (!terminated) 
      return Invalid;
    continue;
  }];
}

// C++-style comment. This rule requires an automata with `loop = 1`
def InlineCommentStx : Rule<"InlineComment"> {
  let rule = [{ '//' }];
  let action = [{
    auto isValid = [](char c){ return c != '\n' && c != '\r'; };
    while (*state && isValid(*(state++)));
    continue;
  }];
}

// Identifier
def IdentifierStx : Rule<"Identifier", "[a-zA-Z_] [a-zA-Z_0-9]*">;

//===----------------------------------------------------------------------===//
// Automatons
//===----------------------------------------------------------------------===//

// Automata for recognizing int literals
def IntAutomata : FiniteAutomata<"Int"> {
  let loop = 0;
  // Definitions required by the IntLiteral class
  let definitions = [
    DigitStx,
    BinaryDigitStx,
    OctalDigitStx,
    HexDigitStx,
    NonZeroStx,
    DigitSeqStx,
    DigitSeqZStx
  ];
  // Rules in the automata
  let rules = !listconcat(
    IntLiteral<"">.rules,
    IntLiteral<"i8">.rules,
    IntLiteral<"i16">.rules,
    IntLiteral<"i32">.rules,
    IntLiteral<"i64">.rules,
    IntLiteral<"u">.rules,
    IntLiteral<"u8">.rules,
    IntLiteral<"u16">.rules,
    IntLiteral<"u32">.rules,
    IntLiteral<"u64">.rules
  );
}

// Automata for recognizing float literals
def FloatAutomata : FiniteAutomata<"Float"> {
  let loop = 0;
  // Definitions required by the FloatLiteral class
  let definitions = [
    DigitStx,
    NonZeroStx,
    DigitSeqStx,
    DigitSeqZStx
  ];
  // Rules in the automata
  let rules = !listconcat(
    FloatLiteral<"">.rules,
    FloatLiteral<"f8">.rules,
    FloatLiteral<"f16">.rules,
    FloatLiteral<"f32">.rules,
    FloatLiteral<"f64">.rules,
    FloatLiteral<"f128">.rules
  );
}

//===----------------------------------------------------------------------===//
// Token classes
//===----------------------------------------------------------------------===//

// Integer literal token class
def IntegerLiteral : TokenClass<"IntLit"> {
  let tokens = IntAutomata.rules;
}

// Signed integer literal token class
def SignedIntegerLiteral : TokenClass<"SignedIntLit"> {
  let tokens = !listconcat(
    IntLiteral<"">.rules,
    IntLiteral<"i8">.rules,
    IntLiteral<"i16">.rules,
    IntLiteral<"i32">.rules,
    IntLiteral<"i64">.rules
  );
}

// Unsigned integer literal token class
def UnsignedIntegerLiteral : TokenClass<"UnsignedIntLit"> {
  let tokens = !listconcat(
    IntLiteral<"u">.rules,
    IntLiteral<"u8">.rules,
    IntLiteral<"u16">.rules,
    IntLiteral<"u32">.rules,
    IntLiteral<"u64">.rules
  );
}

// Floating-point literal token class
def FloatLiteral : TokenClass<"FloatLit"> {
  let tokens = FloatAutomata.rules;
}

#endif
