/* =========================================================================
 *  TOPIC 2 · Compiler for a Starter Language
 * FILE: trace.h   —   Shared tracing helper
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *
 * UNCHANGED SINCE TOPIC 1 — the interfaces held, which is the point
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

#ifndef TRACE_H
#define TRACE_H
#include <stdio.h>
#include <stdarg.h>

/* ----------------------------------------------------------------------------
 * TRACING
 * The compiler narrates what it is doing, because watching a phase run is the
 * fastest way to understand it.  Once the narration stops being useful, pass
 * -q on the command line and every trace() call goes quiet.  Real error
 * messages use fprintf(stderr, ...) and are never suppressed.
 * --------------------------------------------------------------------------*/
extern int quiet;

static inline void trace(const char* fmt, ...) {
    if (quiet) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

#endif
