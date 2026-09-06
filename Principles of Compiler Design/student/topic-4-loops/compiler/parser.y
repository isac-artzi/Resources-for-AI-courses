/* =========================================================================
 *  TOPIC 4 · Compiling Loops
 * FILE: parser.y   —   Phase 2 — Syntax analysis
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *              ^^^^^^  this file
 *
 * WHAT IS NEW IN TOPIC 4
 *   • Relational operators, slotted BELOW + - * / in the precedence table
 *   • `while (cond) stmt` and `for (init; cond; update) stmt`
 *   • `break;` to leave a loop early
 *
 * WHAT COMES NEXT
 *   Topic 5 adds decisions (if, if-else, switch) and the logical operators.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 4)  is yours to write.
 *   Everything else already works — it is the Topic 3 compiler you
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
%token RETURN
/* --------------------------------------------------------------
 * TODO (Topic 4)
 * Declare the loop keywords: WHILE FOR BREAK.
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 4)
 * Declare the two-character relational operator tokens: LE GE EQ NE.
 * (The single-character ones, < and >, arrive as character literals and
 *  need no %token declaration.)
 * -------------------------------------------------------------- */

/* NON-TERMINAL TYPES */
%type <node> program stmt_list stmt decl assign expr print_stmt
%type <node> decl_or_func_list decl_or_func func_def params param_list param
%type <node> return_stmt block func_call arg_list args
/* --------------------------------------------------------------
 * TODO (Topic 4)
 * Every new non-terminal needs its type declared here, or bison will
 * not know that $$ is an ASTNode*.  Add: while_stmt for_stmt
 * for_init for_cond for_update break_stmt
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
/* --------------------------------------------------------------
 * TODO (Topic 4)
 * Give the relational operators their precedence.  They bind LOOSER
 * than + - * /, which is what makes  a + 1 < b * 2  parse as
 *   (a + 1) < (b * 2)  and not  a + (1 < b) * 2.
 * -------------------------------------------------------------- */
%left '+' '-'
%left '*' '/'
%right UMINUS              /* unary minus                            */

%%

/* PROGRAM RULE - Entry point */
program:
    decl_or_func_list {
        root = $1;
    }
    ;

/* Declaration or function list */
decl_or_func_list:
    decl_or_func {
        $$ = $1;
    }
    | decl_or_func_list decl_or_func {
        $$ = createStmtList($1, $2);
    }
    ;

/* Can be either a function definition or a variable declaration */
decl_or_func:
    func_def { $$ = $1; }
    | decl { $$ = $1; }
    ;


/* FUNCTION DEFINITION - int name(params) { body } */
func_def:
    INT ID '(' params ')' block {
        $$ = createFuncDef($2, $4, $6);
        free($2);
    }
    | INT ID '(' ')' block {
        $$ = createFuncDef($2, NULL, $5);
        free($2);
    }
    ;

/* PARAMETERS */
params:
    param_list { $$ = $1; }
    ;

param_list:
    param {
        $$ = $1;
    }
    | param_list ',' param {
        $$ = createParamList($1, $3);
    }
    ;

param:
    INT ID {
        $$ = createParam($2);
        free($2);
    }
    | INT ID '[' ']' {
        $$ = createArrayParam($2);
        free($2);
    }
    ;

/* BLOCK STATEMENT */
block:
    '{' stmt_list '}' {
        $$ = createBlock($2);
    }
    | '{' '}' {
        $$ = createBlock(NULL);
    }
    ;


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
    | return_stmt
/* --------------------------------------------------------------
 * TODO (Topic 4)
 * A loop and a `break` are statements too.  Add them as alternatives:
 *   | while_stmt
 *   | for_stmt
 *   | break_stmt
 * -------------------------------------------------------------- */
    | block
    | func_call ';' { $$ = $1; }
    ;


/* --------------------------------------------------------------
 * TODO (Topic 4)
 * BREAK STATEMENT.  One rule, one action:
 *   break_stmt: BREAK ';' { $$ = createBreak(); } ;
 * -------------------------------------------------------------- */

/* DECLARATION - int x; or int arr[10]; */
decl:
    INT ID ';' {
        $$ = createDecl("int", $2);
        free($2);
    }
    | INT ID '[' NUM ']' ';' {
        $$ = createArrayDecl($2, $4);
        free($2);
    }
    ;

/* ASSIGNMENT - x = expr; or arr[i] = expr; */
assign:
    ID '=' expr ';' {
        $$ = createAssign($1, $3);
        free($1);
    }
    | ID '[' expr ']' '=' expr ';' {
        ASTNode* lhs = createArrayIndex($1, $3);
        $$ = createAssign(NULL, $6);
        $$->data.assign.arrayLHS = lhs;
        free($1);
    }
    ;

/* RETURN STATEMENT */
return_stmt:
    RETURN expr ';' {
        $$ = createReturn($2);
    }
    | RETURN ';' {
        $$ = createReturn(NULL);
    }
    ;


/* --------------------------------------------------------------
 * TODO (Topic 4)
 * WHILE and FOR.
 *
 *   while_stmt : WHILE '(' expr ')' stmt          -> createWhile(cond, body)
 *   for_stmt   : FOR '(' for_init ';' for_cond ';' for_update ')' stmt
 *                                                  -> createFor(init,cond,update,body)
 *
 * All three parts of a for-header are OPTIONAL, so for_init, for_cond
 * and for_update each need an empty alternative that yields NULL.
 * Note what for_init is NOT: it is an assignment WITHOUT a semicolon,
 * because the `;` belongs to the for-header, not to the assignment.
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
    | ID '[' expr ']' {
        $$ = createArrayIndex($1, $3);
        free($1);
    }
    | func_call {
        $$ = $1;
    }
    | expr '+' expr {
        $$ = createBinOp('+', $1, $3);
    }
    | expr '-' expr {
        $$ = createBinOp('-', $1, $3);
    }
    | expr '*' expr {
        $$ = createBinOp('*', $1, $3);
    }
    | expr '/' expr {
        $$ = createBinOp('/', $1, $3);
    }
/* --------------------------------------------------------------
 * TODO (Topic 4)
 * The six relational operators, each producing a BINOP node.
 * createBinOp packs the operator into ONE char, so the two-character
 * operators get single-letter codes:  <= is 'l',  >= is 'g',
 * == is 'e',  != is 'n'.  (See opText() in ast.c for the full table.)
 * A comparison yields 1 or 0 — there is no separate boolean type.
 * -------------------------------------------------------------- */
    | '-' expr %prec UMINUS {
        $$ = createBinOp('u', $2, NULL);  /* 'u' for unary minus */
    }
    | '(' expr ')' {
        $$ = $2;
    }
    ;

/* FUNCTION CALL */
func_call:
    ID '(' args ')' {
        $$ = createFuncCall($1, $3);
        free($1);
    }
    | ID '(' ')' {
        $$ = createFuncCall($1, NULL);
        free($1);
    }
    ;

/* ARGUMENTS */
args:
    arg_list { $$ = $1; }
    ;

arg_list:
    expr {
        $$ = $1;  /* Single argument becomes the arg node */
    }
    | arg_list ',' expr {
        $$ = createArgList($1, $3);
    }
    ;


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
