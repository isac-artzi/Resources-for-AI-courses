/* =========================================================================
 *  TOPIC 3 · Compiling Complex Variables and Functions
 * FILE: semantic.c   —   Phase 3 — Semantic analysis
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                               ^^^^^^^^  this file
 *
 * WHAT IS NEW IN TOPIC 3
 *   • A scope STACK replaces the single flat scope
 *   • Function signatures are collected first, so calls may appear before definitions
 *   • Argument count is checked against the declaration
 *
 * WHAT COMES NEXT
 *   Topic 4 adds loops (while, for, break) and the label/jump machinery they need.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 3)  is yours to write.
 *   Everything else already works — it is the Topic 2 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
 * ========================================================================= */

/* SEMANTIC ANALYSIS IMPLEMENTATION - WITH FUNCTION SUPPORT
 * Performs semantic checks on the Abstract Syntax Tree
 * Now supports functions, scopes, parameters, control flow
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "semantic.h"
#include "symtab.h"
#include "trace.h"

#define MAX_FUNCTIONS 100
#define MAX_PARAMS 20
#define MAX_SCOPE_DEPTH 10

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * A second table, for FUNCTIONS.  Variables and functions live in separate
 * namespaces, which is why a program may declare both `int total;` and
 * `int total(...)`.  Record just enough to check a call: the name, the
 * parameter count, and which parameters are arrays.
 * -------------------------------------------------------------- */

/* Scope for variables */
typedef struct {
    char* names[MAX_VARS];
    int count;
} Scope;

/* Global semantic information */
static SemanticInfo semInfo;
static Scope scopes[MAX_SCOPE_DEPTH];
static int scopeDepth = 0;

/* Initialize semantic analyzer */
void initSemantic() {
    semInfo.errorCount = 0;
    semInfo.warningCount = 0;
    scopeDepth = 0;
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Reset the function table too, and seed it with the one function the
 * language provides for free: `print`, which takes one argument.
 * Registering it here is what makes `print(x)` type-check like any
 * other call instead of needing a special case.
 * -------------------------------------------------------------- */

    trace("SEMANTIC ANALYZER: initialized\n\n");
}

/* Scope management */
static void enterScope() {
    if (scopeDepth >= MAX_SCOPE_DEPTH) {
        fprintf(stderr, "SEMANTIC ERROR: Maximum scope depth exceeded\n");
        semInfo.errorCount++;
        return;
    }
    scopes[scopeDepth].count = 0;
    scopeDepth++;
}

static void exitScope() {
    if (scopeDepth > 0) {
        /* Free variable names in this scope */
        for (int i = 0; i < scopes[scopeDepth - 1].count; i++) {
            free(scopes[scopeDepth - 1].names[i]);
        }
        scopeDepth--;
    }
}

/* Print current semantic scopes for debugging */
static void printSemanticScopes() {
    trace("\n┌─────────────────────────────────────────────────────────┐\n");
    trace("│ SEMANTIC SCOPE STACK (Depth: %d)                        \n", scopeDepth);
    trace("├─────────────────────────────────────────────────────────┤\n");

    if (scopeDepth == 0) {
        trace("│ (no active scopes)                                      │\n");
    } else {
        for (int depth = 0; depth < scopeDepth; depth++) {
            if (depth == 0) {
                trace("│ Scope[%d] GLOBAL (%d variables)                        \n", depth, scopes[depth].count);
            } else {
                trace("│ Scope[%d] LOCAL (%d variables)                         \n", depth, scopes[depth].count);
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Implement this part of the milestone.
 * -------------------------------------------------------------- */
            }

            if (scopes[depth].count > 0) {
                trace("│   Variables: ");
                for (int i = 0; i < scopes[depth].count; i++) {
                    trace("%s", scopes[depth].names[i]);
                    if (i < scopes[depth].count - 1) trace(", ");
                }
                trace("\n");
            } else {
                trace("│   (empty)\n");
            }
        }
    }
    trace("└─────────────────────────────────────────────────────────┘\n\n");
}

/* Add variable to current scope */
/* RESERVED IDENTIFIERS
 * The intermediate-code generator invents names of its own: temporaries
 * t0, t1, ... and labels L0, L1, ...  If a user variable had one of those
 * shapes the back end could not tell them apart, so the language reserves
 * that namespace.  Reporting it here — in the semantic analyzer, with a line
 * number — is far kinder than a mysterious wrong answer at run time. */
static int isReservedName(const char* name) {
    if (!name || !name[0]) return 0;
    if (strncmp(name, "__sw", 4) == 0) return 1;          /* switch temporaries */
    if (name[0] != 't' && name[0] != 'L') return 0;
    if (!name[1]) return 0;
    for (int i = 1; name[i]; i++)
        if (name[i] < '0' || name[i] > '9') return 0;
    return 1;
}

static void reportReserved(const char* name, int lineno) {
    fprintf(stderr, "\n╔════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║ SEMANTIC ERROR - Reserved Identifier                      ║\n");
    fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "  📍 Location: Line %d\n", lineno);
    fprintf(stderr, "  ❌ Error: '%s' is reserved for the compiler's own use\n", name);
    fprintf(stderr, "  📖 Note: names of the form t0, t1, ... are three-address-code\n");
    fprintf(stderr, "           temporaries and L0, L1, ... are generated labels.\n");
    fprintf(stderr, "  💡 Suggestion: rename the variable, for example '%s_' or 'total'\n\n", name);
}

static int addVarToScope(char* name) {
    if (scopeDepth == 0) {
        fprintf(stderr, "SEMANTIC ERROR: No scope to add variable to\n");
        return -1;
    }

    Scope* currentScope = &scopes[scopeDepth - 1];

    /* Check if already declared in current scope */
    for (int i = 0; i < currentScope->count; i++) {
        if (strcmp(currentScope->names[i], name) == 0) {
            return -1;  /* Already declared in this scope */
        }
    }

    /* Add to current scope */
    if (currentScope->count >= MAX_VARS) {
        fprintf(stderr, "SEMANTIC ERROR: Too many variables in scope\n");
        return -1;
    }

    currentScope->names[currentScope->count] = strdup(name);
    currentScope->count++;
    return 0;
}

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Arrays need their own scope-insertion helper — not because the storage
 * question is different (symtab.c settles that later) but because the
 * error messages are.
 * -------------------------------------------------------------- */
/* Check if variable is declared in any visible scope */
static int isVarDeclaredInScope(char* name) {
    /* Search from innermost to outermost scope */
    for (int depth = scopeDepth - 1; depth >= 0; depth--) {
        for (int i = 0; i < scopes[depth].count; i++) {
            if (strcmp(scopes[depth].names[i], name) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * findFunction / addFunction / countParams / countArgs.
 * countParams walks the parameter list parser.y built; countArgs walks
 * the argument list at a call site.  Comparing those two numbers is the
 * whole of call checking.
 * -------------------------------------------------------------- */
/* Forward declaration: checkStmt and checkStmtList are mutually recursive,
 * which is exactly what you want when the thing being checked is a tree. */
static void checkStmtList(ASTNode* node);

/* Check expression for semantic correctness */
static void checkExpr(ASTNode* node) {
    if (!node) return;

    switch(node->type) {
        case NODE_NUM:
            /* Literal numbers are always valid */
            break;

        case NODE_VAR:
            if (!isVarDeclaredInScope(node->data.name)) {
                fprintf(stderr, "\n╔════════════════════════════════════════════════════════════╗\n");
                fprintf(stderr, "║ SEMANTIC ERROR - Undeclared Variable                      ║\n");
                fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
                fprintf(stderr, "  📍 Location: Line %d\n", node->lineno);
                fprintf(stderr, "  ❌ Error: Variable '%s' is used before being declared\n", node->data.name);
                fprintf(stderr, "  💡 Suggestion: Add a declaration before using this variable:\n");
                fprintf(stderr, "     → int %s;  (add this before line %d)\n", node->data.name, node->lineno);
                fprintf(stderr, "  📖 Note: In C-minus, all variables must be declared before use\n\n");
                semInfo.errorCount++;
            }
            break;

        case NODE_BINOP:
            checkExpr(node->data.binop.left);
            if (node->data.binop.right) {  /* Unary minus has no right operand */
                checkExpr(node->data.binop.right);
            }
            break;

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Check a NODE_FUNC_CALL: does the function exist, and does the call
 * supply the right number of arguments?  Then check each argument
 * expression.  A wrong argument count is the single most common mistake
 * a compiler can catch for the programmer here.
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Check a NODE_ARRAY_INDEX: the array must be declared, and the index is
 * an expression that must itself check out.  Note what we do NOT check —
 * whether the index is in range.  That is not knowable at compile time in
 * general, and this language has no run-time check either.  Say so in
 * your write-up; it is a real design decision, not an oversight.
 * -------------------------------------------------------------- */
        default:
            break;
    }
}

/* Check statement */
static void checkStmt(ASTNode* node) {
    if (!node) return;

    switch(node->type) {
        case NODE_DECL:
            if (isReservedName(node->data.decl.name)) {
                reportReserved(node->data.decl.name, node->lineno);
                semInfo.errorCount++;
                break;
            }
            if (addVarToScope(node->data.decl.name) == -1) {
                fprintf(stderr, "\n╔════════════════════════════════════════════════════════════╗\n");
                fprintf(stderr, "║ SEMANTIC ERROR - Duplicate Variable Declaration           ║\n");
                fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
                fprintf(stderr, "  📍 Location: Line %d\n", node->lineno);
                fprintf(stderr, "  ❌ Error: Variable '%s' is already declared in the current scope\n", node->data.decl.name);
                fprintf(stderr, "  💡 Suggestion: Choose a different variable name or remove the duplicate declaration\n");
                fprintf(stderr, "  📖 Note: Each variable can only be declared once per scope\n");
                fprintf(stderr, "     → Use assignment (=) to change the value of an existing variable\n");
                fprintf(stderr, "     → Or use a different name: %s2, my%s, etc.\n\n", node->data.decl.name, node->data.decl.name);
                semInfo.errorCount++;
            } else {
                trace("  ✓ Variable '%s' declared (line %d)\n", node->data.decl.name, node->lineno);
            }
            break;

        case NODE_ASSIGN:
                if (!isVarDeclaredInScope(node->data.assign.var)) {
                    fprintf(stderr, "\n╔════════════════════════════════════════════════════════════╗\n");
                    fprintf(stderr, "║ SEMANTIC ERROR - Assignment to Undeclared Variable        ║\n");
                    fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
                    fprintf(stderr, "  📍 Location: Line %d\n", node->lineno);
                    fprintf(stderr, "  ❌ Error: Trying to assign to variable '%s' which hasn't been declared\n", node->data.assign.var);
                    fprintf(stderr, "  💡 Suggestion: Declare the variable before assigning to it:\n");
                    fprintf(stderr, "     → Add this before line %d:\n", node->lineno);
                    fprintf(stderr, "       int %s;\n", node->data.assign.var);
                    fprintf(stderr, "  📖 Remember: Variables must be declared before use in C-minus\n\n");
                    semInfo.errorCount++;
                } else {
                    trace("  ✓ Assignment to '%s' is valid (line %d)\n", node->data.assign.var, node->lineno);
                }
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * The target of an assignment may now be an array ELEMENT.  Check
 * the index expression in that case, and the variable name in the
 * other.
 * -------------------------------------------------------------- */
            checkExpr(node->data.assign.value);
            break;

        case NODE_PRINT:
            checkExpr(node->data.expr);
            trace("  ✓ Print statement is valid\n");
            break;

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Check a NODE_RETURN.  The interesting case is `return` OUTSIDE any
 * function — legal to parse, meaningless to translate.  That gap between
 * "parses" and "means something" is the territory this phase owns.
 * -------------------------------------------------------------- */

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * NODE_BLOCK opens a scope and closes it again.  Two lines of code, and
 * they are what make this legal:
 *     int x;  { int x; }      inner x shadows outer x
 * and this an error:
 *     { int x; int x; }       same name twice in one scope
 * Handle NODE_FUNC_CALL used as a statement and NODE_STMT_LIST too.
 * -------------------------------------------------------------- */

        case NODE_STMT_LIST:
            /* The grammar is left-recursive, so a statement list can contain
             * another statement list.  Handling that here (not only in
             * checkStmtList) is what lets the recursion bottom out. */
            checkStmtList(node);
            break;

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Check a NODE_ARRAY_DECL.  Two forms to tell apart: a real declaration
 * (which needs a positive size) and a parameter (which has no size at
 * all).  Reject `int a[0];` and `int a[-3];` here — nothing downstream
 * will.
 * -------------------------------------------------------------- */
        default:
            break;
    }
}

/* Check statement list */
static void checkStmtList(ASTNode* node) {
    if (!node) return;

    if (node->type == NODE_STMT_LIST) {
        checkStmt(node->data.stmtlist.stmt);
        checkStmtList(node->data.stmtlist.next);
    } else {
        checkStmt(node);
    }
}

/* --------------------------------------------------------------
 * TODO (Topic 3)
 * addParamsToScope, checkFuncDef, registerFunction, registerFunctions
 * and checkFunctions — the TWO-PASS structure.
 *
 * Pass 1 records every function signature.  Pass 2 checks the bodies.
 * Two passes and not one, so that a function may call another defined
 * LATER in the file.  Collapse them into one pass and see which programs
 * stop compiling; that experiment is worth five minutes.
 * -------------------------------------------------------------- */
int performSemanticAnalysis(ASTNode* root) {
    if (!root) {
        fprintf(stderr, "SEMANTIC ERROR: No AST to analyze\n");
        return -1;
    }

    trace("Running semantic analysis with function support...\n\n");

    /* Enter global scope */
    enterScope();
    trace("Entered global scope\n");
    printSemanticScopes();

    /* The starter language has no functions, so one walk over the statement
     * list is the whole analysis.  Topic 3 replaces this with two passes. */
    checkStmtList(root);
/* --------------------------------------------------------------
 * TODO (Topic 3)
 * Replace the single walk with TWO passes:
 *   pass 1: registerFunctions(root) — record every signature
 *   pass 2: checkFunctions(root)    — check every body
 * -------------------------------------------------------------- */

    /* Exit global scope */
    exitScope();

    return semInfo.errorCount > 0 ? -1 : 0;
}

/* Print semantic analysis summary */
void printSemanticSummary() {
    trace("═══════════════════════════════════════════\n");
    trace("SEMANTIC ANALYSIS SUMMARY\n");
    trace("═══════════════════════════════════════════\n");
    trace("Errors found:       %d\n", semInfo.errorCount);
    trace("Warnings found:     %d\n", semInfo.warningCount);
    trace("\n");

    if (semInfo.errorCount == 0) {
        trace("✓ Semantic analysis passed - program is semantically correct!\n");
    } else {
        trace("✗ Semantic analysis failed - fix errors before proceeding\n");
    }
    trace("═══════════════════════════════════════════\n\n");
}
