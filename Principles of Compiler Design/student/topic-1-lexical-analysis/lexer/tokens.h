#ifndef TOKENS_H
#define TOKENS_H

/* ============================================================================
 *  TOPIC 1 · Compiler Design Phases
 * FILE: tokens.h   —   Phase 1: Lexical analysis (the vocabulary)
 * ----------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *   ^^^^^^^  this milestone is the first box, and only the first box
 *
 * WHAT THIS FILE IS
 * A token is a (KIND, LEXEME, LOCATION) triple:
 *
 *     source text :   count = count + 1 ;
 *     tokens      :   ID(count)  ASSIGN  ID(count)  PLUS  NUM(1)  SEMI
 *
 * The KIND is what the parser will make decisions on in Topic 2.  The LEXEME
 * is the text that matched, which the parser needs for identifiers and
 * numbers and throws away for everything else.  The LOCATION is what turns
 * "syntax error" into "syntax error, line 12, column 7" — the single largest
 * difference between a compiler people can use and one they cannot.
 *
 * WHAT COMES NEXT
 *   Topic 2 replaces the printed token table with a real consumer: bison's
 *   yyparse() calls yylex() one token at a time and builds a syntax tree.
 *   The token KINDS below become bison's %token declarations, and this file
 *   is replaced by the generated parser.tab.h.  Everything else about the
 *   scanner survives unchanged all the way to Topic 6 — which is why it is
 *   worth getting right now.
 * ==========================================================================*/

/* ---------------------------------------------------------------------------
 * TOKEN KINDS
 * These cover the WHOLE language the course builds, not just what Topic 2
 * uses.  Recognizing a keyword the parser cannot yet handle is fine: the
 * scanner's job is vocabulary, and the grammar's job is sentences.
 * -------------------------------------------------------------------------*/
typedef enum {
    TOK_EOF = 0,

    /* Keywords */
    TOK_INT, TOK_PRINT, TOK_RETURN,
    TOK_IF, TOK_ELSE, TOK_WHILE, TOK_FOR,
    TOK_SWITCH, TOK_CASE, TOK_DEFAULT, TOK_BREAK,

    /* Identifiers and literals */
    TOK_ID, TOK_NUM,

    /* Operators */
    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
    TOK_LT, TOK_GT, TOK_LE, TOK_GE, TOK_EQ, TOK_NE,
    TOK_AND, TOK_OR, TOK_NOT,
    TOK_ASSIGN,

    /* Delimiters */
    TOK_SEMI, TOK_COLON, TOK_COMMA,
    TOK_LPAREN, TOK_RPAREN,
    TOK_LBRACE, TOK_RBRACE,
    TOK_LBRACKET, TOK_RBRACKET,

    /* Anything the language does not admit */
    TOK_ERROR
} TokenKind;

/* Printable name for each kind, used by the token listing. */
const char* tokenName(TokenKind k);

/* A short description of what the kind IS, so the listing teaches rather
 * than merely reports. */
const char* tokenCategory(TokenKind k);

/* Counters the scanner maintains and main.c reports. */
extern int lexErrorCount;   /* Characters the language does not admit      */
extern int tokenCount;      /* Tokens successfully recognized              */
extern int lineNo;          /* Current line   (1-based)                    */
extern int colNo;           /* Current column (1-based), tracked by hand   */

/* The scanner entry point that flex generates from scanner.l. */
int yylex(void);
extern char* yytext;
extern FILE* yyin;

/* The lexeme and position of the token yylex() just returned. */
extern char lastLexeme[256];
extern int  lastLine;
extern int  lastCol;

#endif
