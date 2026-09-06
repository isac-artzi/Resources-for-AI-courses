/* =========================================================================
 *  TOPIC 2 · Compiler for a Starter Language
 * FILE: ast.c   —   Phase 2 — Syntax analysis (the tree it builds)
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                        ^^^  this file
 *
 * WHAT IS NEW IN TOPIC 2
 *   • One constructor per node kind, plus a tree printer
 *
 * WHAT COMES NEXT
 *   Topic 3 adds functions, arrays and the rest of arithmetic — and with them, real activation records.
 *
 * YOUR TASK
 *   This is Project 2: the first compiler you build end to end.  Sections
 *   marked  TODO (Topic 2)  are yours.  Everything else — the headers, the
 *   scanner, the driver, the register allocator — is given, because the
 *   point of this project is the six PHASES, not the plumbing between them.
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

/* --------------------------------------------------------------------
 * TODO (Topic 2) — THE REMAINING AST CONSTRUCTORS
 * createNum and createVar above are the pattern.  Every constructor does
 * the same four things:
 *
 *     ASTNode* node = malloc(sizeof(ASTNode));
 *     node->type   = NODE_XXX;          <- which kind of node this is
 *     node->lineno = yylineno;          <- where it came from, for errors
 *     ... store the children ...
 *     return node;
 *
 * Write: createBinOp, createDecl, createAssign, createPrint, createStmtList.
 *
 * strdup every char* you store.  The scanner's buffer is reused for the
 * next token, so a name you merely point at will have changed by the time
 * the semantic analyzer reads it.  That bug looks like the AST being
 * randomly corrupted, and it is one of the hardest to find in this course.
 *
 * createStmtList is the one worth thinking about: it links two statements
 * into a list, and the grammar is LEFT recursive, so $1 is the list so far
 * and $2 is the new statement.  Draw the tree for a three-statement
 * program before you write it.
 * -------------------------------------------------------------------- */


/* Display the AST structure (for debugging and education) */
void printAST(ASTNode* node, int level) {
    /* ----------------------------------------------------------------
     * TODO (Topic 2) — THE TREE PRINTER
     * Print the tree, one node per line, indented two spaces per level.
     *
     *     if (!node) return;
     *     for (int i = 0; i < level; i++) printf("  ");
     *     switch (node->type) { ... one case per node kind ... }
     *
     * Recurse into children with level + 1 so the indentation shows the shape.
     * NODE_STMT_LIST is the exception: print its two children at the SAME
     * level, because a list of statements is a sequence, not a nesting.
     *
     * This function is not decoration.  It is the only window you have into
     * Phase 2, and every bug you hit for the rest of the semester gets
     * diagnosed by staring at its output.  Write it early and make it good.
     * ---------------------------------------------------------------- */
}
