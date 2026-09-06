/* =========================================================================
 *  TOPIC 5 · Compiling Control Flow — Decisions
 * FILE: symtab.c   —   Phases 3 & 6 — Symbol table / storage map
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *
 * UNCHANGED SINCE TOPIC 4 — the interfaces held, which is the point
 *
 * WHAT COMES NEXT
 *   Topic 6 adds no new syntax: it measures, documents and hardens what you have.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 5)  is yours to write.
 *   Everything else already works — it is the Topic 4 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
 * ========================================================================= */

/* ============================================================================
 * SYMBOL TABLE IMPLEMENTATION  —  the storage map
 * ----------------------------------------------------------------------------
 * Every identifier the code generator meets has to be turned into an address.
 * This file is the only place that decides what that address is.
 *
 *   int total;          (at file scope)  ->  label  total   in .data
 *   int i;              (inside main)    ->  0($sp)
 *   int scores[10];     (inside main)    ->  4($sp) .. 40($sp)   (40 bytes)
 *   int sum(int a[])    (parameter)      ->  0($sp) holds a POINTER to a[0]
 *
 * The last line is the one students trip over: an array parameter does not
 * copy the array. It receives the caller's base address, so the callee's slot
 * is one word wide no matter how large the array is. That is why
 * `isParamArray` exists and why addArrayParam reserves 4 bytes, not size*4.
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symtab.h"
#include "trace.h"

/* The current function's activation record, and the program's globals. */
static SymbolTable locals;
static SymbolTable globals;

/* Set to 1 once initGlobalScope has run, so a stray lookup before that is
 * reported clearly instead of reading uninitialized memory. */
static int globalsReady = 0;

/* --------------------------------------------------------------------------
 * Internal: find a name in one specific table.
 * -------------------------------------------------------------------------*/
static Symbol* findIn(SymbolTable* t, const char* name) {
    for (int i = 0; i < t->count; i++) {
        if (strcmp(t->vars[i].name, name) == 0) return &t->vars[i];
    }
    return NULL;
}

/* --------------------------------------------------------------------------
 * Internal: append a symbol to a table, or return NULL if the table is full.
 * -------------------------------------------------------------------------*/
static Symbol* appendTo(SymbolTable* t, const char* name, const char* type) {
    if (t->count >= MAX_VARS) {
        fprintf(stderr,
                "Symbol table overflow: more than %d names in one scope "
                "(while declaring '%s')\n", MAX_VARS, name);
        exit(1);
    }
    Symbol* s      = &t->vars[t->count++];
    s->name        = strdup(name);
    s->type        = strdup(type ? type : "int");
    s->offset      = 0;
    s->isArray     = 0;
    s->arraySize   = 0;
    s->isGlobal    = 0;
    s->isParamArray= 0;
    return s;
}

/* ==========================================================================
 * LOCAL SCOPE — one activation record
 * ========================================================================*/

void initSymTab(void) {
    locals.count      = 0;
    locals.nextOffset = 0;   /* Offsets grow upward from $sp */
}

int addVar(char* name, char* type) {
    if (findIn(&locals, name)) return -1;          /* Duplicate declaration */

    Symbol* s = appendTo(&locals, name, type);
    s->offset = locals.nextOffset;
    locals.nextOffset += 4;                        /* One word per int      */
    return s->offset;
}

int addArray(char* name, int size) {
    if (findIn(&locals, name)) return -1;
    if (size <= 0) size = 1;                       /* Defensive: never 0    */

    Symbol* s    = appendTo(&locals, name, "int");
    s->offset    = locals.nextOffset;
    s->isArray   = 1;
    s->arraySize = size;
    locals.nextOffset += size * 4;                 /* size words, contiguous */
    return s->offset;
}

int addArrayParam(char* name) {
    if (findIn(&locals, name)) return -1;

    Symbol* s        = appendTo(&locals, name, "int");
    s->offset        = locals.nextOffset;
    s->isArray       = 1;
    s->arraySize     = 0;      /* Size is unknown to the callee — by design */
    s->isParamArray  = 1;      /* Slot holds an ADDRESS, not the elements   */
    locals.nextOffset += 4;    /* A pointer is one word, whatever it points to */
    return s->offset;
}

int getLocalBytes(void) {
    return locals.nextOffset;
}

/* ==========================================================================
 * GLOBAL SCOPE — the .data section
 * ========================================================================*/

void initGlobalScope(void) {
    globals.count      = 0;
    globals.nextOffset = 0;    /* Globals are label-addressed; no offsets */
    globalsReady       = 1;
}

int addGlobalVar(char* name, char* type) {
    if (findIn(&globals, name)) return -1;
    Symbol* s   = appendTo(&globals, name, type);
    s->isGlobal = 1;
    return 0;
}

int addGlobalArray(char* name, int size) {
    if (findIn(&globals, name)) return -1;
    if (size <= 0) size = 1;
    Symbol* s    = appendTo(&globals, name, "int");
    s->isGlobal  = 1;
    s->isArray   = 1;
    s->arraySize = size;
    return 0;
}

/* ==========================================================================
 * LOOKUP — locals shadow globals, exactly as the language rules require
 * ========================================================================*/

Symbol* lookupSymbol(const char* name) {
    if (!name) return NULL;
    Symbol* s = findIn(&locals, name);            /* Innermost scope first */
    if (s) return s;
    if (!globalsReady) return NULL;
    return findIn(&globals, name);                /* Then file scope       */
}

int getVarOffset(char* name) {
    Symbol* s = lookupSymbol(name);
    if (!s || s->isGlobal) return -1;             /* Globals have no offset */
    return s->offset;
}

int getArraySize(char* name) {
    Symbol* s = lookupSymbol(name);
    if (!s || !s->isArray) return -1;
    return s->arraySize;
}

int isVarDeclared(char* name) {
    return lookupSymbol(name) != NULL;
}

int isArray(char* name) {
    Symbol* s = lookupSymbol(name);
    return s && s->isArray;
}

int isGlobalSymbol(char* name) {
    Symbol* s = lookupSymbol(name);
    return s && s->isGlobal;
}

/* ==========================================================================
 * TRACING — shown while compiling (pass -q to silence it)
 * ========================================================================*/

static void printOne(const Symbol* s) {
    if (s->isParamArray) {
        trace("    %-14s int[]   %4d($sp)   (by reference: slot holds base address)\n",
               s->name, s->offset);
    } else if (s->isArray && s->isGlobal) {
        trace("    %-14s int[%d]%*s .data label   (%d bytes)\n",
               s->name, s->arraySize, 6, "", s->arraySize * 4);
    } else if (s->isArray) {
        trace("    %-14s int[%d]%*s %4d($sp)   (%d bytes)\n",
               s->name, s->arraySize, 6, "", s->offset, s->arraySize * 4);
    } else if (s->isGlobal) {
        trace("    %-14s %-7s .data label\n", s->name, s->type);
    } else {
        trace("    %-14s %-7s %4d($sp)\n", s->name, s->type, s->offset);
    }
}

void printSymTab(void) {
    trace("\n  ┌─ SYMBOL TABLE ─────────────────────────────────────────────┐\n");

    trace("  │ GLOBALS (.data)                                            │\n");
    if (globals.count == 0) {
        trace("    (none)\n");
    } else {
        for (int i = 0; i < globals.count; i++) printOne(&globals.vars[i]);
    }

    trace("  │ LOCALS (current activation record, %3d bytes)              │\n",
           locals.nextOffset);
    if (locals.count == 0) {
        trace("    (none)\n");
    } else {
        for (int i = 0; i < locals.count; i++) printOne(&locals.vars[i]);
    }
    trace("  └────────────────────────────────────────────────────────────┘\n\n");
}
