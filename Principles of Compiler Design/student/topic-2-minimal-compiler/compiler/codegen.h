/* =========================================================================
 *  TOPIC 2 · Compiler for a Starter Language
 * FILE: codegen.h   —   Phase 6 — MIPS code generation
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                                                  ^^^^^^^  this file
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

#ifndef CODEGEN_H
#define CODEGEN_H

#include "ast.h"
#include "tac.h"

/* ============================================================================
 * PHASE 6 — TARGET CODE GENERATION (MIPS32, runnable under SPIM / QtSPIM)
 * ----------------------------------------------------------------------------
 * Input : the OPTIMIZED three-address code produced by tac.c
 * Output: a .s file that SPIM can load and run
 *
 * Two problems have to be solved here, and they are the two problems every
 * real back end solves:
 *
 *   1. STORAGE.  TAC pretends it has unlimited named locations
 *      (a, b, arr, t0, t1, ...).  A CPU has 32 registers and a stack.
 *      We give every name a permanent home in memory (see symtab.c) and
 *      treat the registers as a *cache* over those homes.
 *
 *   2. CONTROL.  TAC has functions, calls and returns as single instructions.
 *      MIPS has only `jal` and `jr`.  Everything else — saving the return
 *      address, making room for locals, putting arguments where the callee
 *      expects them, tearing the frame back down — is our job.  That bundle
 *      of conventions is the ACTIVATION RECORD, and it is laid out below.
 *
 * ACTIVATION RECORD (one per function call, built by the prologue)
 *
 *        high address
 *        ┌──────────────────────────────┐
 *        │ saved $ra                    │  frameSize-4($sp)
 *        ├──────────────────────────────┤
 *        │ (reserved / alignment)       │  frameSize-8($sp)
 *        ├──────────────────────────────┤
 *        │ parameters, locals, arrays,  │  0($sp) .. localBytes-1($sp)
 *        │ and compiler temporaries     │
 *        └──────────────────────────────┘  <- $sp
 *        low address
 *
 * Every name a function uses gets a slot in that middle region, so ANY value
 * can always be written back to memory.  That single guarantee is what makes
 * register allocation, spilling and calls safe.
 *
 * CALLING CONVENTION (a deliberate subset of the real MIPS o32 ABI)
 *   $a0-$a3 : first four arguments (arrays are passed as a base ADDRESS)
 *   $v0     : return value
 *   $ra     : return address, saved in the frame so calls can nest and recurse
 *   $t0-$t9 : caller-saved scratch.  Everything live is written back to its
 *             home slot before `jal`, so the callee may clobber all of them.
 * ==========================================================================*/

#define NUM_TEMP_REGS 10   /* $t0 - $t9                                     */
#define MAX_VAR_NAME  64
#define MAX_ARGS      4    /* $a0-$a3; more than this is a reported error   */

/* ---------------------------------------------------------------------------
 * REGISTER DESCRIPTOR — what a machine register currently holds.
 * The allocator is a write-back cache: `dirty` means the register holds a
 * newer value than the memory home, so it must be stored before eviction.
 * -------------------------------------------------------------------------*/
typedef struct {
    char varName[MAX_VAR_NAME]; /* Name cached here ("" when free)          */
    int  isDirty;               /* 1 = register newer than memory           */
    int  inUse;                 /* 1 = currently allocated                  */
    int  hasHome;               /* 0 = literal/scratch, nothing to write back*/
    int  lastUsed;              /* Timestamp, for LRU victim selection      */
} RegisterDescriptor;

typedef struct {
    RegisterDescriptor regs[NUM_TEMP_REGS];
    int timestamp;              /* Monotonic clock for LRU                  */
    int spillCount;             /* How many spills the program needed       */
    int loadCount;              /* How many reloads from memory             */
} RegisterAllocator;

/* Register allocator lifecycle (exposed so main.c can report statistics) */
void initRegAlloc(void);
void printRegAllocStats(void);

/* The one entry point: translate the optimized TAC into a MIPS .s file. */
void generateMIPSFromTAC(const char* filename);

#endif
