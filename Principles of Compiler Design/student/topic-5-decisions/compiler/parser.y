/* =========================================================================
 *  TOPIC 5 · Compiling Control Flow — Decisions
 * FILE: parser.y   —   Phase 2 — Syntax analysis
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *              ^^^^^^  this file
 *
 * WHAT IS NEW IN TOPIC 5
 *   • `if (cond) stmt` and `if (cond) stmt else stmt`
 *   • The dangling-else ambiguity resolved with %nonassoc precedence
 *   • `switch` with `case`, `default` and fall-through
 *   • Logical operators, placed at the BOTTOM of the precedence table
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
%token WHILE FOR BREAK
%token LE GE EQ NE
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * Declare the decision keywords and logical operators:
 *   IF ELSE SWITCH CASE DEFAULT AND OR
 * -------------------------------------------------------------- */

/* NON-TERMINAL TYPES */
%type <node> program stmt_list stmt decl assign expr print_stmt
%type <node> decl_or_func_list decl_or_func func_def params param_list param
%type <node> return_stmt block func_call arg_list args
%type <node> while_stmt for_stmt for_init for_cond for_update break_stmt
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * Add the decision non-terminals: if_stmt switch_stmt case_list
 * case_clause opt_stmt_list
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
 * TODO (Topic 5)
 * Resolve the dangling-else ambiguity and give the logical operators
 * their precedence.  Four lines, and their ORDER is the whole answer:
 *   %nonassoc LOWER_THAN_ELSE
 *   %nonassoc ELSE
 *   %left OR
 *   %left AND
 * Ask yourself why OR must be listed BEFORE AND, and what would break
 * if ELSE were listed before LOWER_THAN_ELSE.
 * -------------------------------------------------------------- */
%left EQ NE
%left '<' '>' LE GE
%left '+' '-'
%left '*' '/'
%right UMINUS              /* unary minus                            */
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * Unary ! binds at least as tightly as unary minus.  Declare it:
 *   %right NOT
 * -------------------------------------------------------------- */

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
    | while_stmt
    | for_stmt
    | break_stmt
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * Add the decision statements: if_stmt and switch_stmt.
 * -------------------------------------------------------------- */
    | block
    | func_call ';' { $$ = $1; }
    ;

/* --------------------------------------------------------------
 * TODO (Topic 5)
 * SWITCH STATEMENT.
 *
 *   switch_stmt : SWITCH '(' expr ')' '{' case_list '}'
 *   case_list   : (empty) | case_list case_clause
 *   case_clause : CASE NUM ':' opt_stmt_list | DEFAULT ':' opt_stmt_list
 *   opt_stmt_list : (empty) | stmt_list
 *
 * case_list builds a LINKED LIST of clauses in source order — append to
 * the tail, do not prepend, because fall-through depends on the order.
 * A case body may be empty; that is how `case 4: case 5: ...` works.
 * -------------------------------------------------------------- */

/* BREAK STATEMENT — leave the innermost loop (Topic 5: or switch) early */
break_stmt:
    BREAK ';' {
        $$ = createBreak();
    }
    ;

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
 * TODO (Topic 5)
 * IF STATEMENT — two forms, and the dangling-else problem.
 *
 *   if_stmt : IF '(' expr ')' stmt %prec LOWER_THAN_ELSE
 *           | IF '(' expr ')' stmt ELSE stmt
 *
 * Build them with createIf(cond, thenStmt, elseStmt); pass NULL for
 * elseStmt in the first form.  The %prec marker is what tells bison to
 * attach a trailing `else` to the NEAREST `if`.  Build it once without
 * the marker and read the conflict report — that report is the lesson.
 * -------------------------------------------------------------- */

/* WHILE LOOP */
while_stmt:
    WHILE '(' expr ')' stmt {
        $$ = createWhile($3, $5);
    }
    ;

/* FOR LOOP - for (init; condition; update) body
 * All three header parts are optional (may be empty).
 * Syntax: for ( for_init ; for_cond ; for_update ) stmt
 */
for_stmt:
    FOR '(' for_init ';' for_cond ';' for_update ')' stmt {
        $$ = createFor($3, $5, $7, $9);
    }
    ;

/* FOR INIT - optional initialization (assignment without semicolon) */
for_init:
    /* empty - no initialization */  {
        $$ = NULL;
    }
    | ID '=' expr {
        /* Simple scalar assignment: e.g., i = 0 */
        $$ = createAssign($1, $3);
        free($1);
    }
    | ID '[' expr ']' '=' expr {
        /* Array element assignment: e.g., arr[0] = 0 */
        ASTNode* lhs = createArrayIndex($1, $3);
        $$ = createAssign(NULL, $6);
        $$->data.assign.arrayLHS = lhs;
        free($1);
    }
    ;

/* FOR CONDITION - optional loop condition expression */
for_cond:
    /* empty - no condition means loop forever (always true) */ {
        $$ = NULL;
    }
    | expr {
        /* Condition expression: e.g., i < 10 */
        $$ = $1;
    }
    ;

/* FOR UPDATE - optional per-iteration update (assignment without semicolon) */
for_update:
    /* empty - no update step */ {
        $$ = NULL;
    }
    | ID '=' expr {
        /* Simple scalar update: e.g., i = i + 1 */
        $$ = createAssign($1, $3);
        free($1);
    }
    | ID '[' expr ']' '=' expr {
        /* Array element update: e.g., arr[i] = arr[i] + 1 */
        ASTNode* lhs = createArrayIndex($1, $3);
        $$ = createAssign(NULL, $6);
        $$->data.assign.arrayLHS = lhs;
        free($1);
    }
    ;


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
    | expr '<' expr {
        $$ = createBinOp('<', $1, $3);
    }
    | expr '>' expr {
        $$ = createBinOp('>', $1, $3);
    }
    | expr LE expr {
        $$ = createBinOp('l', $1, $3);  /* 'l' for <= */
    }
    | expr GE expr {
        $$ = createBinOp('g', $1, $3);  /* 'g' for >= */
    }
    | expr EQ expr {
        $$ = createBinOp('e', $1, $3);  /* 'e' for == */
    }
    | expr NE expr {
        $$ = createBinOp('n', $1, $3);  /* 'n' for != */
    }
    | '-' expr %prec UMINUS {
        $$ = createBinOp('u', $2, NULL);  /* 'u' for unary minus */
    }
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * The three logical operators.  Codes: '&' for &&, '|' for ||, '!' for !.
 * ! is unary, so pass NULL as the right operand — exactly like unary minus.
 * -------------------------------------------------------------- */
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
