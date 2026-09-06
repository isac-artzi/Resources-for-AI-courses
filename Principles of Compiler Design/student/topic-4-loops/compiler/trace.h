/* =========================================================================
 *  TOPIC 4 · Compiling Loops
 * FILE: trace.h   —   Shared tracing helper
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
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
