/* =========================================================================
 *  TOPIC 4 · Compiling Loops
 * FILE: semantic.h   —   Phase 3 — Semantic analysis
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                               ^^^^^^^^  this file
 *
 * UNCHANGED SINCE TOPIC 3 — the interfaces held, which is the point
 *
 * WHAT COMES NEXT
 *   Topic 5 adds decisions (if, if-else, switch) and the logical operators.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 4)  is yours to write.
 *   Everything else already works — it is the Topic 3 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
 * ========================================================================= */

#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "ast.h"

/* SEMANTIC ANALYSIS - WITH FUNCTION SUPPORT
 * This phase checks the semantic correctness of the program
 * Ensures variables are declared before use, no redeclarations, etc.
 * Now supports functions, scopes, parameters, and control flow
 */

/* SEMANTIC ERROR TRACKING */
typedef struct {
    int errorCount;           /* Number of semantic errors found */
    int warningCount;         /* Number of warnings issued */
} SemanticInfo;

/* SEMANTIC ANALYSIS FUNCTIONS */
void initSemantic();                     /* Initialize semantic analyzer */
int performSemanticAnalysis(ASTNode* root);  /* Run semantic checks on AST, returns 0 if successful */
void printSemanticSummary();             /* Print summary of semantic analysis results */

#endif
