/* =========================================================================
 *  TOPIC 3 · Compiling Complex Variables and Functions
 * FILE: parser.y   —   Phase 2 — Syntax analysis
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *              ^^^^^^  this file
 *
 * WHAT IS NEW IN TOPIC 3
 *   • A program is now a list of declarations AND function definitions
 *   • Array declaration `int a[10];`, indexing `a[i]`, and array parameters `int a[]`
 *   • Full arithmetic with precedence: + - * / , parentheses, unary minus
 *   • Function definitions, parameter lists, calls, and `return`
 *
 * WHAT COMES NEXT
 *   Topic 4 adds loops (while, for, break) and the label/jump machinery they need.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 3)  is yours to write.
 *   Everything else already works — it is the Topic 2 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
 * ========================================================================= */

%{
/* PHASE 2 — SYNTAX ANALYSIS
 *
 * The parser answers one question: do these tokens form a legal program?
 * Bison builds an LALR(1) bottom-up parser from the grammar below.  It reads
 * tokens left to right, shifts them onto a stack, and REDUCES whenever the
 * top of the stack matches the right-hand side of a rule.
 *
 * Answering "yes" is not enough, though — the rest of the compiler needs the
 * program's STRUCTURE.  So each rule carries a semantic action in { } that
 * builds one node of the abstract syntax tree.  $1, $2, ... are the values of
 * the symbols on the right-hand side; $$ is the value this rule hands back.
 *
 *      expr '+' expr   { $$ = createBinOp('+', $1, $3); }
 *       │        │                              │   └── right operand
 *       │        └── $3                         └────── left operand
 *       └── $1
 *
 * By the time yyparse() returns, the tree has been built bottom-up beneath us
 * and `root` points at the whole program.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"

/* External declarations for lexer interface */
extern int yylex();
extern int yyparse();
extern FILE* yyin;
extern int yylineno;  /* Line number from scanner */

void yyerror(const char* s);
ASTNode* root = NULL;
%}

/* SEMANTIC VALUES UNION */
%union {
    int num;
    char* str;
    struct ASTNode* node;
}

/* TOKEN DECLARATIONS */
%token <num> NUM
%token <str> ID
%token INT PRINT
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Functions need one more keyword token: RETURN.
 * -------------------------------------------------------------- */

/* NON-TERMINAL TYPES */
%type <node> program stmt_list stmt decl assign expr print_stmt
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Declare the types of the new non-terminals, or bison will not know
 * that their $$ is an ASTNode*:
 *   decl_or_func_list decl_or_func func_def params param_list param
 *   return_stmt block func_call arg_list args
 * -------------------------------------------------------------- */

/* OPERATOR PRECEDENCE AND ASSOCIATIVITY
 * Listed from lowest to highest precedence.
 *
 * DANGLING-ELSE RESOLUTION:
 *   The grammar has an ambiguity: in  if (c) if (c) s  else s
 *   the "else" can bind to either "if".  The standard resolution
 *   is that else binds to the nearest (innermost) if.
 *
 *   We express this by giving the if-without-else rule a lower
 *   precedence (%prec LOWER_THAN_ELSE) than the ELSE token, so
 *   when bison sees ELSE it shifts (attaches it to the inner if)
 *   rather than reducing (closing the outer if first).
 */
%left '+' '-'
%left '*' '/'
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Implement this part of the milestone.
 * -------------------------------------------------------------- */

%%

/* PROGRAM RULE — in the starter language a program IS a list of statements.
 * Topic 3 replaces this rule the moment functions arrive. */
program:
    stmt_list { root = $1; }
    ;
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * A program is no longer a flat list of statements.  It is a list of
 * top-level DECLARATIONS and FUNCTION DEFINITIONS:
 *   program           : decl_or_func_list
 *   decl_or_func_list : decl_or_func | decl_or_func_list decl_or_func
 *   decl_or_func      : func_def | decl
 * Executable statements now live only inside a function body — which
 * is why `main` suddenly matters.
 * -------------------------------------------------------------- */

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * FUNCTION DEFINITIONS, PARAMETERS and BLOCKS.
 *   func_def   : INT ID '(' params ')' block  |  INT ID '(' ')' block
 *   param_list : param | param_list ',' param
 *   param      : INT ID           (a scalar parameter)
 *              | INT ID '[' ']'   (an array parameter — note: no size)
 *   block      : '{' stmt_list '}' | '{' '}'
 * An array parameter carries no size because arrays are passed by
 * REFERENCE: the callee receives an address, not a copy of the data.
 * -------------------------------------------------------------- */

/* STATEMENT LIST */
stmt_list:
    stmt {
        $$ = $1;
    }
    | stmt_list stmt {
        $$ = createStmtList($1, $2);
    }
    ;

/* STATEMENT TYPES */
stmt:
    decl
    | assign
    | print_stmt
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * A `return`, a `{ ... }` block, and a bare function call used for its
 * effect are all statements now.  Add three alternatives:
 *   | return_stmt      | block      | func_call ';'
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Implement this part of the milestone.
 * -------------------------------------------------------------- */
    ;


/* DECLARATION - int x; or int arr[10]; */
decl:
    INT ID ';' {
        $$ = createDecl("int", $2);
        free($2);
    }
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Array declaration:  int arr[10];
 * The size is a NUM, not an expression — the compiler must know how many
 * bytes to reserve, and it must know that at compile time.
 * -------------------------------------------------------------- */
    ;

/* ASSIGNMENT - x = expr; or arr[i] = expr; */
assign:
    ID '=' expr ';' {
        $$ = createAssign($1, $3);
        free($1);
    }
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Assignment to an array ELEMENT:  arr[i] = expr;
 * The left-hand side is now a tree, not just a name, so the assignment node
 * stores it in arrayLHS and leaves `var` NULL.
 * -------------------------------------------------------------- */
    ;

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * RETURN, in both forms:  return expr;  and  return;
 * -------------------------------------------------------------- */


/* EXPRESSION RULES */
expr:
    NUM {
        $$ = createNum($1);
    }
    | ID {
        $$ = createVar($1);
        free($1);
    }
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Two new kinds of primary expression: an array element  arr[i]  and a
 * function call used for its value.
 * -------------------------------------------------------------- */
    | expr '+' expr {
        $$ = createBinOp('+', $1, $3);
    }
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * The rest of arithmetic.  Do NOT write precedence into these rules — the
 * %left declarations above already did that.  Rewriting the grammar into
 * term/factor layers to get precedence is the classic wrong turn here;
 * it works, but it is the answer to a question bison already answered.
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Unary minus.  %prec UMINUS is what stops  -2 * 3  parsing as  -(2 * 3):
 * without it the rule inherits the precedence of BINARY '-', which is
 * lower than '*'.
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Parentheses.  Note there is no node for them: ( e ) just yields e.
 * Parentheses exist to steer the PARSER; once the tree is built the
 * grouping they asked for is recorded in its shape and they vanish.
 * -------------------------------------------------------------- */
    ;

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * FUNCTION CALLS and their arguments.
 *   func_call : ID '(' args ')' | ID '(' ')'
 *   args      : (empty) | arg_list
 *   arg_list  : expr | arg_list ',' expr
 * Build the list with createArgList so tac.c can walk it in source
 * order — argument order is part of the calling convention.
 * -------------------------------------------------------------- */

/* PRINT STATEMENT */
print_stmt:
    PRINT '(' expr ')' ';' {
        $$ = createPrint($3);
    }
    ;

%%

/* ERROR HANDLING */
void yyerror(const char* s) {
    fprintf(stderr, "Syntax Error at line %d: %s\n", yylineno, s);
}
