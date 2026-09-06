/* =========================================================================
 *  TOPIC 5 · Compiling Control Flow — Decisions
 * FILE: tac.h   —   Phases 4 & 5 — Intermediate code and optimization
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                                           ^^^  this file
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

#ifndef TAC_H
#define TAC_H

#include <stddef.h>
#include "ast.h"

/* THREE-ADDRESS CODE (TAC) - WITH FUNCTION SUPPORT
 * Intermediate representation between AST and machine code
 * Each instruction has at most 3 operands (result = arg1 op arg2)
 * Now supports functions, control flow, and all operators
 */

/* TAC INSTRUCTION TYPES
 *
 * The whole instruction set is declared here from the start, even though the
 * front end does not emit all of it yet.  That is deliberate: the IR is the
 * CONTRACT between the front end and the back end, and a contract that keeps
 * changing shape is no contract at all.  The note against each group says
 * which milestone starts producing it.
 */
typedef enum {
    /* Arithmetic operations */
    TAC_ADD,        /* result = arg1 + arg2 */
    TAC_SUB,        /* result = arg1 - arg2 */
    TAC_MUL,        /* result = arg1 * arg2 */
    TAC_DIV,        /* result = arg1 / arg2 */
    TAC_NEG,        /* result = -arg1 (unary minus) */

    /* Comparison operations — emitted from Topic 4 (loops need conditions) */
    TAC_LT,         /* result = arg1 < arg2 */
    TAC_GT,         /* result = arg1 > arg2 */
    TAC_LE,         /* result = arg1 <= arg2 */
    TAC_GE,         /* result = arg1 >= arg2 */
    TAC_EQ,         /* result = arg1 == arg2 */
    TAC_NE,         /* result = arg1 != arg2 */

    /* Logical operations — emitted from Topic 5.  Operands are truthy/falsy;
     * the result is always exactly 0 or 1. */
    TAC_AND,        /* result = arg1 && arg2 */
    TAC_OR,         /* result = arg1 || arg2 */
    TAC_NOT,        /* result = !arg1 */

    /* Assignment and I/O */
    TAC_ASSIGN,     /* result = arg1 */
    TAC_PRINT,      /* print(arg1) */
    TAC_DECL,       /* declare result */

    /* Control flow — emitted from Topic 4.  Every loop, branch and switch in
     * the source language is lowered to these four instructions. */
    TAC_LABEL,      /* label: */
    TAC_GOTO,       /* goto label */
    TAC_IF_FALSE,   /* if_false arg1 goto label */
    TAC_IF_TRUE,    /* if_true arg1 goto label */

    /* Function operations — emitted from Topic 3 */
    TAC_FUNC_BEGIN, /* function name: */
    TAC_FUNC_END,   /* end function name */
    TAC_PARAM,      /* param name */
    TAC_ARG,        /* arg value */
    TAC_CALL,       /* result = call name, argCount */
    TAC_RETURN,     /* return arg1 */

    /* Array operations — emitted from Topic 3 */
    TAC_ARRAY_DECL, /* declare array: result[arg1] */
    TAC_ARRAY_LOAD, /* result = array[index]: result = arg1[arg2] */
    TAC_ARRAY_STORE /* array[index] = value: arg1[arg2] = result */
} TACOp;

/* TAC INSTRUCTION STRUCTURE */
typedef struct TACInstr {
    TACOp op;               /* Operation type */
    char* arg1;             /* First operand */
    char* arg2;             /* Second operand */
    char* result;           /* Result/destination */
    struct TACInstr* next;  /* Linked list pointer */
} TACInstr;

/* TAC LIST MANAGEMENT */
typedef struct {
    TACInstr* head;    /* First instruction */
    TACInstr* tail;    /* Last instruction */
    int tempCount;     /* Counter for temporary variables */
    int labelCount;    /* Counter for labels */
} TACList;

/* TEMPORARY VARIABLE ALLOCATION/DEALLOCATION */
#define MAX_TEMPS 100

typedef struct {
    int allocated[MAX_TEMPS];
    int maxUsed;
    int freeList[MAX_TEMPS];
    int freeCount;
} TempAllocator;

/* TAC GENERATION FUNCTIONS */
void initTAC();                                                    /* Initialize TAC */
char* newTemp();                                                   /* Generate new temp */
char* allocTemp();                                                 /* Allocate temp with reuse */
void freeTemp(char* temp);                                         /* Free temp */
char* newLabel();                                                  /* Generate new label */
void printTempAllocatorState();                                    /* Display stats */

TACInstr* createTAC(TACOp op, char* arg1, char* arg2, char* result); /* Create TAC */
void appendTAC(TACInstr* instr);                                  /* Add to list */
void generateTAC(ASTNode* node);                                  /* Generate from AST */
char* generateTACExpr(ASTNode* node);                             /* Generate for expr */

/* OPTIMIZATION STATISTICS
 * Recorded by optimizeTAC and reported by main.c.  Topic 4 asks students to
 * quantify the gain from optimization; these are the raw numbers to quote. */
typedef struct {
    int passes;              /* Passes run before reaching a fixed point   */
    int instructionsBefore;  /* TAC instruction count before optimization  */
    int instructionsAfter;   /* ... and after                              */
    int algebraic;           /* x+0, x*1, x*0, x/1 simplifications         */
    int constFold;           /* Operations on two literals evaluated       */
    int constProp;           /* Variables replaced by a known constant     */
    int copyProp;            /* Copies replaced by their source            */
    int deadCode;            /* Assignments removed as never read          */
    int unreachable;         /* Instructions removed as unreachable        */
    int branch;              /* Constant-condition branches simplified     */
} OptStats;

/* TAC OPTIMIZATION AND OUTPUT */
void printTAC(void);                        /* Display unoptimized listing        */
void optimizeTAC(void);                     /* Run optimization to a fixed point  */
void printOptimizedTAC(void);               /* Display optimized listing          */
TACList* getOptimizedTAC(void);             /* The list the back end consumes     */
TACList* getUnoptimizedTAC(void);           /* The list before optimization       */
OptStats getOptStats(void);                 /* What optimization accomplished     */
int  countTAC(const TACList* list);         /* Instruction count (a size metric)  */
void formatTAC(const TACInstr* i, char* buf, size_t n);  /* Render one instruction */
void saveTACToFile(const char* filename);
void saveOptimizedTACToFile(const char* filename);

#endif
