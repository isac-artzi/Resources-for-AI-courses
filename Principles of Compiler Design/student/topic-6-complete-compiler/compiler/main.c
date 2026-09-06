/* =========================================================================
 *  TOPIC 6 · Compiler Design and Implementation
 * FILE: main.c   —   The driver — runs all six phases in order
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *
 * WHAT IS NEW IN TOPIC 6
 *   • The full metrics report: phase timings, optimization scorecard, spill counts
 *
 * WHAT COMES NEXT
 *   This is the final milestone.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 6)  is yours to write.
 *   Everything else already works — it is the Topic 5 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
 * ========================================================================= */

/* ============================================================================
 * THE DRIVER — the compiler's main(), and a map of the whole course
 * ----------------------------------------------------------------------------
 * Everything this program does happens in six phases, in this order:
 *
 *   1. LEXICAL ANALYSIS      scanner.l    characters  -> tokens
 *   2. SYNTAX ANALYSIS       parser.y     tokens      -> abstract syntax tree
 *   3. SEMANTIC ANALYSIS     semantic.c   AST         -> AST + a verdict
 *   4. INTERMEDIATE CODE     tac.c        AST         -> three-address code
 *   5. OPTIMIZATION          tac.c        TAC         -> smaller, faster TAC
 *   6. CODE GENERATION       codegen.c    TAC         -> MIPS assembly
 *
 * Each phase gets its own file, its own header, and its own section below.
 * If you can say which phase a bug lives in, you have already done most of
 * the work of fixing it.
 *
 * USAGE
 *   ./minicompiler <input.c> <output.s> [-q]
 *
 *   -q   quiet: print only errors and the final summary.  Use this once the
 *        trace stops being useful and you just want the assembly.
 *
 * SIDE OUTPUTS (named after <output.s>)
 *   output.tac            the intermediate code before optimization
 *   output.optimized.tac  the intermediate code the back end actually used
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ast.h"
#include "semantic.h"
#include "codegen.h"
#include "tac.h"

extern int yyparse(void);
extern FILE* yyin;
extern ASTNode* root;

int quiet = 0;   /* Set by -q; consulted by the banner helpers below */

/* --------------------------------------------------------------------------
 * Phase timing.  Topic 4 asks you to quantify what optimization buys; Topic 6
 * asks for compilation-time metrics.  Both need numbers, so the driver takes
 * them for every phase rather than bolting instrumentation on later.
 * -------------------------------------------------------------------------*/
#define NUM_PHASES 6
static const char* phaseName[NUM_PHASES] = {
    "1. Lexical + syntax analysis",
    "2. AST construction",
    "3. Semantic analysis",
    "4. Intermediate code (TAC)",
    "5. Optimization",
    "6. MIPS code generation"
};
static double phaseMs[NUM_PHASES];
static clock_t phaseStart;

static void beginPhase(int n) {
    phaseStart = clock();
    if (quiet) return;
    printf("\n┌──────────────────────────────────────────────────────────────┐\n");
    printf("│ PHASE %-54s │\n", phaseName[n]);
    printf("└──────────────────────────────────────────────────────────────┘\n");
}
static void endPhase(int n) {
    phaseMs[n] = 1000.0 * (double)(clock() - phaseStart) / CLOCKS_PER_SEC;
}

/* Derive the .tac filenames from the requested .s filename. */
static void deriveTacNames(const char* out, char* tac, char* opt) {
    const char* dot   = strrchr(out, '.');
    const char* slash = strrchr(out, '/');
    size_t base = (dot && (!slash || dot > slash)) ? (size_t)(dot - out) : strlen(out);
    snprintf(tac, 256, "%.*s.tac", (int)base, out);
    snprintf(opt, 256, "%.*s.optimized.tac", (int)base, out);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input.c> <output.s> [-q]\n", argv[0]);
        printf("Example: ./minicompiler test.c test.s\n");
        return 1;
    }
    for (int i = 3; i < argc; i++)
        if (strcmp(argv[i], "-q") == 0) quiet = 1;

    yyin = fopen(argv[1], "r");
    if (!yyin) {
        fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
        return 1;
    }

    if (!quiet) {
        printf("\n╔══════════════════════════════════════════════════════════════╗\n");
        printf("║  MINI COMPILER                                               ║\n");
        printf("║  source: %-51s ║\n", argv[1]);
        printf("║  target: %-51s ║\n", argv[2]);
        printf("╚══════════════════════════════════════════════════════════════╝\n");
    }

    clock_t totalStart = clock();

    /* ---------------- PHASES 1 & 2: scanning and parsing ------------------
     * Bison drives Flex: yyparse() asks yylex() for one token at a time and
     * reduces them according to the grammar.  The semantic actions in
     * parser.y build the AST as the reductions happen, so by the time
     * yyparse() returns 0 the tree is already standing. */
    beginPhase(0);
    int parseFailed = yyparse();
    endPhase(0);

    if (parseFailed != 0 || root == NULL) {
        printf("\n✗ Compilation stopped: the program is not syntactically valid.\n");
        printf("  Fix the errors listed above and compile again.\n");
        fclose(yyin);
        return 1;
    }
    if (!quiet) printf("✓ Parse succeeded — the program matches the grammar.\n");

    beginPhase(1);
    if (!quiet) {
        printf("The AST is the program with the punctuation thrown away:\n");
        printf("only the structure that later phases actually need survives.\n\n");
        printAST(root, 0);
    }
    endPhase(1);

    /* ---------------- PHASE 3: semantic analysis --------------------------
     * Syntax says the sentence is well formed.  Semantics says it means
     * something: names are declared before use, calls match their
     * definitions, break appears only where it can break out of something. */
    beginPhase(2);
    initSemantic();
    int semErrors = performSemanticAnalysis(root);
    printSemanticSummary();
    endPhase(2);

    if (semErrors != 0) {
        printf("\n✗ Compilation stopped: the program is syntactically valid but\n");
        printf("  does not mean anything the compiler can translate.  See the\n");
        printf("  semantic errors listed above.\n");
        fclose(yyin);
        return 1;
    }

    /* ---------------- PHASE 4: intermediate code --------------------------
     * TAC is the hinge of the compiler.  Above it, everything is about the
     * source language; below it, everything is about the target machine.
     * Retargeting to ARM means rewriting only what is below this line. */
    beginPhase(3);
    initTAC();
    generateTAC(root);
    if (!quiet) printTAC();
    char tacFile[256], optTacFile[256];
    deriveTacNames(argv[2], tacFile, optTacFile);
    saveTACToFile(tacFile);
    endPhase(3);

    /* ---------------- PHASE 5: optimization ------------------------------- */
    beginPhase(4);
    optimizeTAC();
    if (!quiet) printOptimizedTAC();
    saveOptimizedTACToFile(optTacFile);
    endPhase(4);

    /* ---------------- PHASE 6: MIPS code generation ------------------------ */
    beginPhase(5);
    generateMIPSFromTAC(argv[2]);
    endPhase(5);

    double totalMs = 1000.0 * (double)(clock() - totalStart) / CLOCKS_PER_SEC;

    /* ---------------- REPORT ---------------------------------------------- */
    OptStats os = getOptStats();
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  COMPILATION SUCCESSFUL                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  Assembly written to : %s\n", argv[2]);
    printf("  TAC listings        : %s, %s\n", tacFile, optTacFile);
    printf("\n  Optimization\n");
    printf("    passes to fixed point : %d\n", os.passes);
    printf("    TAC instructions      : %d -> %d",
           os.instructionsBefore, os.instructionsAfter);
    if (os.instructionsBefore)
        printf("  (%.1f%% smaller)",
               100.0 * (os.instructionsBefore - os.instructionsAfter) / os.instructionsBefore);
    printf("\n");
    printf("\n  Code generation\n");
    printRegAllocStats();

    printf("\n  Compilation time by phase\n");
    for (int i = 0; i < NUM_PHASES; i++)
        printf("    %-30s %7.2f ms\n", phaseName[i], phaseMs[i]);
    printf("    %-30s %7.2f ms\n", "TOTAL", totalMs);

    printf("\n  Next step: run it.\n");
    printf("    spim -file %s        (or open %s in QtSPIM)\n\n", argv[2], argv[2]);

    fclose(yyin);
    return 0;
}
