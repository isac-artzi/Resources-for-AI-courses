/* =========================================================================
 *  TOPIC 6 · Compiler Design and Implementation
 * FILE: scanner.l   —   Phase 1 — Lexical analysis
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *   ^^^^^^^  this file
 *
 * UNCHANGED SINCE TOPIC 5 — the interfaces held, which is the point
 * (last changed in Topic 5: Keywords `if`, `else`, `switch`, `case`, `default`)
 *
 * WHAT COMES NEXT
 *   This is the final milestone.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 6)  is yours to write.
 *   Everything else already works — it is the Topic 5 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
 * ========================================================================= */

%{
/* PHASE 1 — LEXICAL ANALYSIS
 *
 * The scanner turns a stream of CHARACTERS into a stream of TOKENS.
 * It is the only part of the compiler that ever sees whitespace or comments.
 *
 *      int  sum  =  a + 10 ;
 *      └─┬┘ └─┬┘ │  │ │ └┬┘ │
 *       INT  ID  '=' ID '+' NUM ';'
 *
 * Flex turns the rules below into a finite automaton.  Two rules decide
 * everything you need to know about how it behaves:
 *
 *   LONGEST MATCH   "<=" wins over "<" because it matches more characters.
 *                   This is why "<=" must exist as a rule at all, and why
 *                   its position in the file does not matter.
 *   FIRST MATCH     when two rules match the SAME length, the one written
 *                   first wins.  This is why the keyword rules sit above
 *                   the identifier rule: without that order, "int" would
 *                   come back as an ID.
 */
#include <stdio.h>
#include <stdlib.h>
#include "parser.tab.h"  /* Token definitions from Bison */
%}

/* Flex options to suppress warnings for unused functions */
%option nounput
%option noinput
%option yylineno

%%

"int"           { return INT; }
"print"         { return PRINT; }
"return"        { return RETURN; }
"if"            { return IF; }
"else"          { return ELSE; }
"while"         { return WHILE; }
"for"           { return FOR; }
"switch"        { return SWITCH; }
"case"          { return CASE; }
"default"       { return DEFAULT; }
"break"         { return BREAK; }

[a-zA-Z_][a-zA-Z0-9_]* {
    yylval.str = strdup(yytext);
    return ID;
}

[0-9]+          {
    yylval.num = atoi(yytext);
    return NUM;
}

"&&"            { return AND; }
"||"            { return OR; }

"<="            { return LE; }
">="            { return GE; }
"=="            { return EQ; }
"!="            { return NE; }

"+"             { return '+'; }
"-"             { return '-'; }
"*"             { return '*'; }
"/"             { return '/'; }

"<"             { return '<'; }
">"             { return '>'; }
"!"             { return '!'; }

"="             { return '='; }

";"             { return ';'; }
":"             { return ':'; }
"("             { return '('; }
")"             { return ')'; }
"{"             { return '{'; }
"}"             { return '}'; }
"["             { return '['; }
"]"             { return ']'; }
","             { return ','; }

"//".*          { /* Skip single-line comments */ }
"/*"([^*]|\*+[^*/])*\*+"/"  { /* Skip multi-line comments */ }

[ \t\n]         { }

.               { printf("Lexical Error: Unknown character '%c'\n", *yytext); }

%%

int yywrap() {
    return 1;
}
