/* ============================================================================
 *  TOPIC 1 · Compiler Design Phases
 * FILE: main.c   —   the driver for Phase 1
 *
 * YOUR TASK
 *   Two functions below are marked TODO (Topic 1).  Everything else in this
 *   file already works: read it before you write anything, because it shows
 *   you exactly what your scanner has to supply.
 * ----------------------------------------------------------------------------
 * This program is a compiler with exactly one phase in it.  It reads a source
 * file, asks the scanner for tokens until there are none left, and prints
 * what it got.  That is all a lexical analyzer ever does.
 *
 * The output is deliberately a TABLE rather than a running commentary,
 * because the table is the thing you hand to the parser in Topic 2 and the
 * thing you point at in your video when you say "here is what my scanner
 * produces for this input".
 *
 *   ./lexer program.cm          list every token
 *   ./lexer program.cm -c       count only (no listing)
 *
 * EXIT STATUS
 *   0   no lexical errors — the source is certified lexically correct
 *   1   at least one lexical error, all of them reported above
 *   2   the file could not be opened
 *
 * A non-zero exit status matters: it is what lets `make` and a test script
 * tell success from failure without reading the output.
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokens.h"

/* --------------------------------------------------------------------------
 * The printable name of each token kind.  Keeping this next to the enum in
 * tokens.h is a maintenance trap — add a kind there and forget it here and
 * the table silently prints "?".  A real compiler solves this with an X-macro;
 * doing that is a good optional exercise.
 * -------------------------------------------------------------------------*/
const char* tokenName(TokenKind k) {
    /* ----------------------------------------------------------------
     * TODO (Topic 1)
     * Return the printable name of each token kind: TOK_INT -> "INT",
     * TOK_ID -> "ID", TOK_LE -> "LE", and so on for every kind in
     * tokens.h.  A switch with one case per kind is the obvious way.
     *
     * Compile with -Wall and let the compiler tell you which kinds you
     * forgot: a switch over an enum with a missing case is a warning.
     * ---------------------------------------------------------------- */
    (void)k;
    return "?";
}

const char* tokenCategory(TokenKind k) {
    /* ----------------------------------------------------------------
     * TODO (Topic 1)
     * Return what the kind IS, so the table explains itself:
     *   keyword · identifier · integer literal · arithmetic operator
     *   relational operator · logical operator · assignment · delimiter
     * Group the cases; there are only eight answers for thirty-odd kinds.
     * ---------------------------------------------------------------- */
    (void)k;
    return "?";
}

int main(int argc, char** argv) {
    int countOnly = 0;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <source-file> [-c]\n", argv[0]);
        fprintf(stderr, "  -c   print only the summary, not the token table\n");
        return 2;
    }
    for (int i = 2; i < argc; i++)
        if (strcmp(argv[i], "-c") == 0) countOnly = 1;

    yyin = fopen(argv[1], "r");
    if (!yyin) {
        fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 2;
    }

    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  PHASE 1 · LEXICAL ANALYSIS                            ║\n");
    printf("║  source: %-55s ║\n", argv[1]);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    if (!countOnly) {
        printf("   #  LINE  COL  TOKEN        LEXEME            KIND\n");
        printf(" ───  ────  ───  ───────────  ────────────────  ─────────────────────\n");
    }

    int n = 0;
    int kind;
    while ((kind = yylex()) != 0) {
        n++;
        if (!countOnly)
            printf(" %3d  %4d  %3d  %-11s  %-16.16s  %s\n",
                   n, lastLine, lastCol,
                   tokenName((TokenKind)kind), lastLexeme,
                   tokenCategory((TokenKind)kind));
    }

    fclose(yyin);

    printf("\n───────────────────────────────────────────────────────────────────\n");
    printf("  Tokens recognized : %d\n", tokenCount);
    printf("  Lines read        : %d\n", lineNo);
    printf("  Lexical errors    : %d\n", lexErrorCount);
    printf("───────────────────────────────────────────────────────────────────\n");

    if (tokenCount == 0) {
        printf("\n  ⚠  NO TOKENS RECOGNISED\n");
        printf("     The scanner read the file and produced nothing.  That is\n");
        printf("     not a certificate of correctness — it means no rule in\n");
        printf("     scanner.l matched anything yet.\n\n");
        return 1;
    }

    if (lexErrorCount == 0) {
        printf("\n  ✓ CERTIFIED LEXICALLY CORRECT\n");
        printf("    Every character in this file belongs to the language.\n");
        printf("    That is a claim about VOCABULARY only — this program says\n");
        printf("    nothing about whether the tokens form a legal sentence.\n");
        printf("    Answering that is the parser's job, in Topic 2.\n\n");
        return 0;
    }

    printf("\n  ✗ %d LEXICAL ERROR(S)\n", lexErrorCount);
    printf("    Each one is reported above with its line and column.\n");
    printf("    Scanning continued past every error on purpose, so that one\n");
    printf("    run reports every problem rather than only the first.\n\n");
    return 1;
}
