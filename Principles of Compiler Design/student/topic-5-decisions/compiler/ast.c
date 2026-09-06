/* =========================================================================
 *  TOPIC 5 · Compiling Control Flow — Decisions
 * FILE: ast.c   —   Phase 2 — Syntax analysis (the tree it builds)
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                        ^^^  this file
 *
 * WHAT IS NEW IN TOPIC 5
 *   • Constructors and printing for the decision nodes
 *
 * WHAT COMES NEXT
 *   Topic 6 adds no new syntax: it measures, documents and hardens what you have.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 5)  is yours to write.
 *   Everything else already works — it is the Topic 4 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
 * ========================================================================= */

/* AST IMPLEMENTATION
 * Functions to create and manipulate Abstract Syntax Tree nodes
 * The AST is built during parsing and used for all subsequent phases
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"

/* External line number from scanner */
extern int yylineno;

/* Create a number literal node */
ASTNode* createNum(int value) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_NUM;
    node->lineno = yylineno;
    node->data.num = value;  /* Store the integer value */
    return node;
}

/* Create a variable reference node */
ASTNode* createVar(char* name) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_VAR;
    node->lineno = yylineno;
    node->data.name = strdup(name);  /* Copy the variable name */
    return node;
}

/* Create a binary operation node (for addition) */
ASTNode* createBinOp(char op, ASTNode* left, ASTNode* right) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_BINOP;
    node->lineno = yylineno;
    node->data.binop.op = op;        /* Store operator (+) */
    node->data.binop.left = left;    /* Left subtree */
    node->data.binop.right = right;  /* Right subtree */
    return node;
}

/* Create a variable declaration node */
ASTNode* createDecl(char* type, char* name) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_DECL;
    node->lineno = yylineno;
    node->data.decl.name = strdup(name);
    node->data.decl.varType = strdup(type);
    return node;
}

/* Create an assignment statement node */
ASTNode* createAssign(char* var, ASTNode* value) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_ASSIGN;
    node->lineno = yylineno;
    node->data.assign.var = var ? strdup(var) : NULL;  /* Variable name (NULL for array assignments) */
    node->data.assign.value = value;      /* Expression tree */
    /* No array subscript on the left unless the parser says so */
    node->data.assign.arrayLHS = NULL;
    return node;
}

/* Create a print statement node */
ASTNode* createPrint(ASTNode* expr) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_PRINT;
    node->lineno = yylineno;
    node->data.expr = expr;  /* Expression to print */
    return node;
}

/* Create a statement list node (links statements together) */
ASTNode* createStmtList(ASTNode* stmt1, ASTNode* stmt2) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_STMT_LIST;
    node->lineno = stmt1 ? stmt1->lineno : (stmt2 ? stmt2->lineno : yylineno);
    node->data.stmtlist.stmt = stmt1;  /* First statement */
    node->data.stmtlist.next = stmt2;  /* Rest of list */
    return node;
}

/* Create a function definition node */
ASTNode* createFuncDef(char* name, ASTNode* params, ASTNode* body) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_FUNC_DEF;
    node->lineno = yylineno;
    node->data.func_def.name = strdup(name);
    node->data.func_def.params = params;
    node->data.func_def.body = body;
    return node;
}

/* Create a parameter node */
ASTNode* createParam(char* name) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_PARAM;
    node->lineno = yylineno;
    node->data.param.name = strdup(name);
    return node;
}

/* Create a parameter list node */
ASTNode* createParamList(ASTNode* param, ASTNode* next) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_PARAM_LIST;
    node->lineno = param ? param->lineno : yylineno;
    node->data.param_list.param = param;
    node->data.param_list.next = next;
    return node;
}

/* Create a function call node */
ASTNode* createFuncCall(char* name, ASTNode* args) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_FUNC_CALL;
    node->lineno = yylineno;
    node->data.func_call.name = strdup(name);
    node->data.func_call.args = args;
    return node;
}

/* Create an argument list node */
ASTNode* createArgList(ASTNode* expr, ASTNode* next) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_ARG_LIST;
    node->lineno = expr ? expr->lineno : yylineno;
    node->data.arg_list.expr = expr;
    node->data.arg_list.next = next;
    return node;
}

/* Create a return statement node */
ASTNode* createReturn(ASTNode* expr) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_RETURN;
    node->lineno = yylineno;
    node->data.ret.expr = expr;
    return node;
}

/* --------------------------------------------------------------
 * TODO (Topic 5)
 * createIf: allocate a NODE_IF and store condition, then-branch and
 * else-branch (NULL when the source had no `else`).
 * -------------------------------------------------------------- */
/* Create a while loop node */
ASTNode* createWhile(ASTNode* condition, ASTNode* body) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_WHILE;
    node->lineno = yylineno;
    node->data.while_stmt.condition = condition;
    node->data.while_stmt.body = body;
    return node;
}

/* Create a for loop node
 * Represents: for (init; condition; update) body
 * Any of init, condition, or update may be NULL.
 */
ASTNode* createFor(ASTNode* init, ASTNode* condition, ASTNode* update, ASTNode* body) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_FOR;
    node->lineno = yylineno;
    node->data.for_stmt.init      = init;       /* Initialization (run once before loop) */
    node->data.for_stmt.condition = condition;  /* Condition (checked before each iteration) */
    node->data.for_stmt.update    = update;     /* Update (run after each iteration) */
    node->data.for_stmt.body      = body;       /* Loop body */
    return node;
}

/* Create a block statement node */
ASTNode* createBlock(ASTNode* stmt_list) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_BLOCK;
    node->lineno = yylineno;
    node->data.block.stmt_list = stmt_list;
    return node;
}

/* Create an array declaration node */
ASTNode* createArrayDecl(char* name, int size) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_ARRAY_DECL;
    node->lineno = yylineno;
    node->data.array_decl.name = strdup(name);
    node->data.array_decl.size = size;
    node->data.array_decl.isParam = 0;
    return node;
}

/* Create an array parameter node (size unknown) */
ASTNode* createArrayParam(char* name) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_ARRAY_DECL;
    node->lineno = yylineno;
    node->data.array_decl.name = strdup(name);
    node->data.array_decl.size = 0;  /* Size unknown for parameters */
    node->data.array_decl.isParam = 1;
    return node;
}

/* --------------------------------------------------------------
 * TODO (Topic 5)
 * createSwitch and createCase: allocate NODE_SWITCH / NODE_CASE.
 * A new case clause always starts with next = NULL; parser.y links
 * the clauses together in source order.
 * -------------------------------------------------------------- */
/* Create a break statement node */
ASTNode* createBreak(void) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_BREAK;
    node->lineno = yylineno;
    return node;
}

/* Create an array indexing node */
ASTNode* createArrayIndex(char* name, ASTNode* index) {
    ASTNode* node = malloc(sizeof(ASTNode));
    node->type = NODE_ARRAY_INDEX;
    node->lineno = yylineno;
    node->data.array_index.name = strdup(name);
    node->data.array_index.index = index;
    return node;
}


/* Render an operator code as the text the programmer actually wrote.
 * The parser packs multi-character operators into single chars so that a
 * BINOP node stays small; this is where that encoding is undone. */
const char* opText(char op) {
    switch (op) {
        case '+': return "+";   case '-': return "-";
        case '*': return "*";   case '/': return "/";
        case '<': return "<";   case '>': return ">";
        case 'l': return "<=";  case 'g': return ">=";
        case 'e': return "==";  case 'n': return "!=";
        case 'u': return "unary -";
        case '&': return "&&";  case '|': return "||";  case '!': return "!";
        default:  return "?";
    }
}

/* Display the AST structure (for debugging and education) */
void printAST(ASTNode* node, int level) {
    if (!node) return;

    /* Indent based on tree depth */
    for (int i = 0; i < level; i++) printf("  ");

    /* Print node based on its type */
    switch(node->type) {
        case NODE_NUM:
            printf("NUM: %d\n", node->data.num);
            break;
        case NODE_VAR:
            printf("VAR: %s\n", node->data.name);
            break;
        case NODE_BINOP:
            printf("BINOP: %s\n", opText(node->data.binop.op));
            printAST(node->data.binop.left, level + 1);
            printAST(node->data.binop.right, level + 1);
            break;
        case NODE_DECL:
            printf("DECL: %s %s\n", node->data.decl.varType, node->data.decl.name);
            break;
        case NODE_ASSIGN:
            if (node->data.assign.arrayLHS) {
                printf("ASSIGN (array element)\n");
                for (int i = 0; i < level + 1; i++) printf("  ");
                printf("LHS:\n");
                printAST(node->data.assign.arrayLHS, level + 2);
            } else {
                printf("ASSIGN: %s\n", node->data.assign.var);
            }
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("VALUE:\n");
            printAST(node->data.assign.value, level + 2);
            break;
        case NODE_PRINT:
            printf("PRINT\n");
            printAST(node->data.expr, level + 1);
            break;
        case NODE_STMT_LIST:
            /* Print statements in sequence at same level */
            printAST(node->data.stmtlist.stmt, level);
            printAST(node->data.stmtlist.next, level);
            break;
        case NODE_FUNC_DEF:
            printf("FUNC_DEF: %s\n", node->data.func_def.name);
            if (node->data.func_def.params) {
                for (int i = 0; i < level + 1; i++) printf("  ");
                printf("PARAMS:\n");
                printAST(node->data.func_def.params, level + 2);
            }
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("BODY:\n");
            printAST(node->data.func_def.body, level + 2);
            break;
        case NODE_PARAM:
            printf("PARAM: %s\n", node->data.param.name);
            break;
        case NODE_PARAM_LIST:
            printAST(node->data.param_list.param, level);
            printAST(node->data.param_list.next, level);
            break;
        case NODE_FUNC_CALL:
            printf("FUNC_CALL: %s\n", node->data.func_call.name);
            if (node->data.func_call.args) {
                for (int i = 0; i < level + 1; i++) printf("  ");
                printf("ARGS:\n");
                printAST(node->data.func_call.args, level + 2);
            }
            break;
        case NODE_ARG_LIST:
            printAST(node->data.arg_list.expr, level);
            printAST(node->data.arg_list.next, level);
            break;
        case NODE_RETURN:
            printf("RETURN\n");
            if (node->data.ret.expr) {
                printAST(node->data.ret.expr, level + 1);
            }
            break;
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * Print a NODE_IF: label it, then recurse into condition, then-branch
 * and (if present) else-branch, each indented one level deeper.
 * -------------------------------------------------------------- */
        case NODE_WHILE:
            printf("WHILE\n");
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("CONDITION:\n");
            printAST(node->data.while_stmt.condition, level + 2);
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("BODY:\n");
            printAST(node->data.while_stmt.body, level + 2);
            break;
        case NODE_FOR:
            /* Print the for loop with all three header components */
            printf("FOR\n");
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("INIT:\n");
            if (node->data.for_stmt.init) {
                printAST(node->data.for_stmt.init, level + 2);
            } else {
                for (int i = 0; i < level + 2; i++) printf("  ");
                printf("(empty)\n");
            }
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("CONDITION:\n");
            if (node->data.for_stmt.condition) {
                printAST(node->data.for_stmt.condition, level + 2);
            } else {
                for (int i = 0; i < level + 2; i++) printf("  ");
                printf("(always true)\n");
            }
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("UPDATE:\n");
            if (node->data.for_stmt.update) {
                printAST(node->data.for_stmt.update, level + 2);
            } else {
                for (int i = 0; i < level + 2; i++) printf("  ");
                printf("(empty)\n");
            }
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("BODY:\n");
            printAST(node->data.for_stmt.body, level + 2);
            break;
        case NODE_BLOCK:
            printf("BLOCK\n");
            printAST(node->data.block.stmt_list, level + 1);
            break;
        case NODE_ARRAY_DECL:
            if (node->data.array_decl.isParam) {
                printf("ARRAY_PARAM: %s[]\n", node->data.array_decl.name);
            } else {
                printf("ARRAY_DECL: %s[%d]\n",
                       node->data.array_decl.name,
                       node->data.array_decl.size);
            }
            break;
        case NODE_ARRAY_INDEX:
            printf("ARRAY_INDEX: %s\n", node->data.array_index.name);
            for (int i = 0; i < level + 1; i++) printf("  ");
            printf("INDEX:\n");
            printAST(node->data.array_index.index, level + 2);
            break;
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * Print NODE_SWITCH and NODE_CASE.  Walk the clause list via ->next.
 * -------------------------------------------------------------- */
        case NODE_BREAK:
            printf("BREAK\n");
            break;
    }
}