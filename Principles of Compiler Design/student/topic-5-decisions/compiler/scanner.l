/* =========================================================================
 *  TOPIC 5 · Compiling Control Flow — Decisions
 * FILE: scanner.l   —   Phase 1 — Lexical analysis
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *   ^^^^^^^  this file
 *
 * WHAT IS NEW IN TOPIC 5
 *   • Keywords `if`, `else`, `switch`, `case`, `default`
 *   • Logical operators `&&`, `||`, `!` and the `:` delimiter
 *
 * WHAT COMES NEXT
 *   Topic 6 adds no new syntax: it measures, documents and hardens what you have.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 5)  is yours to write.
 *   Everything else already works — it is the Topic 4 compiler you
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
    /* --------------------------------------------------------------
     * TODO (Topic 5)
     * Return the IF and ELSE tokens for the `if` and `else` keywords.
     * One line each, exactly like the `int` and `print` rules above.
     * -------------------------------------------------------------- */
"while"         { return WHILE; }
"for"           { return FOR; }
    /* --------------------------------------------------------------
     * TODO (Topic 5)
     * Return SWITCH, CASE and DEFAULT for the switch keywords.
     * -------------------------------------------------------------- */
"break"         { return BREAK; }

[a-zA-Z_][a-zA-Z0-9_]* {
    yylval.str = strdup(yytext);
    return ID;
}

[0-9]+          {
    yylval.num = atoi(yytext);
    return NUM;
}

    /* --------------------------------------------------------------
     * TODO (Topic 5)
     * Return AND and OR for the logical operators && and ||.
     * Think about longest match: does `&&` need its own rule to beat anything?
     * -------------------------------------------------------------- */

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
    /* --------------------------------------------------------------
     * TODO (Topic 5)
     * Return the logical-not operator.
     * -------------------------------------------------------------- */

"="             { return '='; }

";"             { return ';'; }
    /* --------------------------------------------------------------
     * TODO (Topic 5)
     * Return ':' — needed to write `case 3:` and `default:`.
     * -------------------------------------------------------------- */
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
