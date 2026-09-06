/* =========================================================================
 *  TOPIC 3 · Compiling Complex Variables and Functions
 * FILE: ast.c   —   Phase 2 — Syntax analysis (the tree it builds)
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                        ^^^  this file
 *
 * WHAT IS NEW IN TOPIC 3
 *   • Constructors and printing for every new node type
 *
 * WHAT COMES NEXT
 *   Topic 4 adds loops (while, for, break) and the label/jump machinery they need.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 3)  is yours to write.
 *   Everything else already works — it is the Topic 2 compiler you
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

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Constructors for the function-related nodes: createFuncDef,
 * createParam, createParamList, createFuncCall, createArgList,
 * createReturn.  They are all the same three lines — malloc, set
 * the type and lineno, store the children — which is the point:
 * the AST is deliberately boring so the phases above it can be
 * interesting.
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * createBlock, createArrayDecl, createArrayParam and
 * createArrayIndex.  Give an array PARAMETER size 0 and isParam 1;
 * give a declared array its real size and isParam 0.  Code
 * generation later depends on telling those two apart.
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * createArrayIndex: build the node for  arr[i].  It stores the array
 * NAME and the index EXPRESSION — note the name is not itself an
 * expression node, which is a simplification worth mentioning in your
 * write-up (it is what stops this language having pointers).
 * -------------------------------------------------------------- */

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
            printf("ASSIGN: %s\n", node->data.assign.var);
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * An assignment now has two shapes: to a variable, or to an array
 * element.  Print them differently — a tree dump that hides the
 * difference is a tree dump you cannot debug with.
 * -------------------------------------------------------------- */
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
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Print the function-related nodes.  Indent children one level
 * deeper so the printed tree really looks like a tree.
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Print NODE_BLOCK, NODE_ARRAY_DECL and NODE_ARRAY_INDEX.
 * -------------------------------------------------------------- */
    }
}