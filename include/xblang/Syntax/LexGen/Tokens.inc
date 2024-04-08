#ifndef TERMINAL
#define TERMINAL(...)
#endif

#ifndef KEYWORD
#define KEYWORD(...) TERMINAL(__VA_ARGS__)
#endif

KEYWORD(EmptyString, "eps")

TERMINAL(Int)
TERMINAL(Hex)
TERMINAL(Code)
TERMINAL(String)
TERMINAL(Identifier)
TERMINAL(Comment)
TERMINAL(Char)

#ifndef PUNCTUATION
#define PUNCTUATION(...) TERMINAL(__VA_ARGS__)
#endif

#ifndef CHAR_PUNCTUATION
#define CHAR_PUNCTUATION(...) PUNCTUATION(__VA_ARGS__)
#endif

CHAR_PUNCTUATION(BOr, '|')
CHAR_PUNCTUATION(Plus, '+')
CHAR_PUNCTUATION(Multiply, '*')
CHAR_PUNCTUATION(Dot, '.')
CHAR_PUNCTUATION(Question, '?')
CHAR_PUNCTUATION(LParen, '(')
CHAR_PUNCTUATION(RParen, ')')
CHAR_PUNCTUATION(LBracket, '[')
CHAR_PUNCTUATION(RBracket, ']')
CHAR_PUNCTUATION(Dash, '-')
CHAR_PUNCTUATION(Quote, '\'')

#ifdef CHAR_PUNCTUATION
#undef CHAR_PUNCTUATION
#endif

#ifdef PUNCTUATION
#undef PUNCTUATION
#endif

#ifdef KEYWORD
#undef KEYWORD
#endif

#ifdef TERMINAL
#undef TERMINAL
#endif
