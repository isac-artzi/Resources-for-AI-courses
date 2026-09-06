/* =========================================================================
 *  TOPIC 6 · Compiler Design and Implementation
 * FILE: parser.y   —   Phase 2 — Syntax analysis
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *              ^^^^^^  this file
 *
 * UNCHANGED SINCE TOPIC 5 — the interfaces held, which is the point
 * (last changed in Topic 5: `if (cond) stmt` and `if (cond) stmt else stmt`)
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
%token IF ELSE SWITCH CASE DEFAULT
%token AND OR

/* NON-TERMINAL TYPES */
%type <node> program stmt_list stmt decl assign expr print_stmt
%type <node> decl_or_func_list decl_or_func func_def params param_list param
%type <node> return_stmt block func_call arg_list args
%type <node> while_stmt for_stmt for_init for_cond for_update break_stmt
%type <node> if_stmt switch_stmt case_list case_clause opt_stmt_list

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
%nonassoc LOWER_THAN_ELSE  /* pseudo-token: lower prec than ELSE */
%nonassoc ELSE             /* ELSE has higher prec → shifts over */
%left OR                   /* lowest: a || b || c groups left        */
%left AND                  /* && binds tighter than ||               */
%left EQ NE
%left '<' '>' LE GE
%left '+' '-'
%left '*' '/'
%right UMINUS              /* unary minus                            */
%right NOT                 /* highest: unary !                       */

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
    | if_stmt
    | switch_stmt
    | block
    | func_call ';' { $$ = $1; }
    ;

/* SWITCH STATEMENT - switch (expr) { case_list } */
switch_stmt:
    SWITCH '(' expr ')' '{' case_list '}' {
        $$ = createSwitch($3, $6);
    }
    ;

/* CASE LIST - zero or more case/default clauses */
case_list:
    /* empty */ {
        $$ = NULL;
    }
    | case_list case_clause {
        /* Append case_clause to end of the linked list */
        if ($1 == NULL) {
            $$ = $2;
        } else {
            ASTNode* tail = $1;
            while (tail->data.case_clause.next)
                tail = tail->data.case_clause.next;
            tail->data.case_clause.next = $2;
            $$ = $1;
        }
    }
    ;

/* CASE CLAUSE - case N: stmts  or  default: stmts */
case_clause:
    CASE NUM ':' opt_stmt_list {
        $$ = createCase($2, 0, $4);
    }
    | DEFAULT ':' opt_stmt_list {
        $$ = createCase(0, 1, $3);
    }
    ;

/* OPTIONAL STATEMENT LIST - body of a case clause (may be empty) */
opt_stmt_list:
    /* empty */ {
        $$ = NULL;
    }
    | stmt_list {
        $$ = $1;
    }
    ;


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

/* IF STATEMENT
 * Two forms:
 *   if (expr) stmt              - no else clause
 *   if (expr) stmt else stmt    - with else clause
 *
 * %prec LOWER_THAN_ELSE on the first rule tells bison that when
 * the lookahead is ELSE this rule has lower precedence, so bison
 * shifts the ELSE and attaches it to the innermost if (correct
 * dangling-else behavior).  Without this marker bison still
 * resolves the conflict the same way but reports it as a warning.
 */
if_stmt:
    IF '(' expr ')' stmt %prec LOWER_THAN_ELSE {
        /* if-without-else: else_stmt is NULL */
        $$ = createIf($3, $5, NULL);
    }
    | IF '(' expr ')' stmt ELSE stmt {
        /* if-with-else: captures the else branch */
        $$ = createIf($3, $5, $7);
    }
    ;


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
    | expr AND expr {
        $$ = createBinOp('&', $1, $3);    /* '&' for logical AND (&&) */
    }
    | expr OR expr {
        $$ = createBinOp('|', $1, $3);    /* '|' for logical OR (||)  */
    }
    | '!' expr %prec NOT {
        $$ = createBinOp('!', $2, NULL);  /* '!' for logical NOT      */
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
