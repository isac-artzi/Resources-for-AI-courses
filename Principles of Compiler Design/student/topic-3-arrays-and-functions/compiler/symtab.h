/* =========================================================================
 *  TOPIC 3 · Compiling Complex Variables and Functions
 * FILE: symtab.h   —   Phases 3 & 6 — Symbol table / storage map
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *
 * WHAT IS NEW IN TOPIC 3
 *   • Split into a global table (.data) and a per-function table (the frame)
 *   • Arrays record their element count; array parameters record that the slot holds an address
 *
 * WHAT COMES NEXT
 *   Topic 4 adds loops (while, for, break) and the label/jump machinery they need.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 3)  is yours to write.
 *   Everything else already works — it is the Topic 2 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
 * ========================================================================= */

#ifndef SYMTAB_H
#define SYMTAB_H

/* ============================================================================
 * SYMBOL TABLE  —  code-generation view of program storage
 * ----------------------------------------------------------------------------
 * There are TWO symbol tables in this compiler, and it is important not to
 * confuse them:
 *
 *   1. semantic.c keeps a *scope stack* used only to answer the question
 *      "is this name visible here?"  It disappears once analysis is done.
 *
 *   2. THIS table is the *storage map*.  It answers the question
 *      "where does this name live at run time?"  Every entry records a
 *      concrete address: a label in .data for globals, or a byte offset
 *      inside the current activation record for locals and parameters.
 *
 * Two tables are kept here because a MIPS program has two storage classes:
 *
 *      GLOBALS  ->  .data section, addressed by label      (la $t0, counter)
 *      LOCALS   ->  the current stack frame, addressed by  (lw $t0, 8($sp))
 *
 * The local table is CLEARED at the start of every function (initSymTab),
 * because each function gets a fresh activation record.  The global table is
 * built once, before any function is generated, and persists for the whole
 * program (initGlobalScope).
 * ========================================================================== */

#define MAX_VARS 128   /* Maximum symbols per scope (locals or globals) */

/* ---------------------------------------------------------------------------
 * SYMBOL ENTRY
 * -------------------------------------------------------------------------*/
typedef struct {
    char* name;        /* Identifier as written in the source                */
    char* type;        /* Declared type — always "int" in this language      */
    int   offset;      /* Locals: byte offset from $sp. Globals: unused (0)  */
    int   isArray;     /* 1 = array, 0 = plain scalar                        */
    int   arraySize;   /* Element count (0 for scalars and array parameters) */
    int   isGlobal;    /* 1 = lives in .data, 0 = lives in the stack frame   */
    int   isParamArray;/* 1 = array PARAMETER: the slot holds a base ADDRESS,
                        *     not the array data itself.  Arrays are passed
                        *     by reference, so `int a[]` receives a pointer. */
} Symbol;

/* ---------------------------------------------------------------------------
 * SYMBOL TABLE — a flat array is plenty for a teaching compiler.
 * A production compiler would use a hash table; the interface below would
 * not change, which is exactly why the interface is worth isolating.
 * -------------------------------------------------------------------------*/
typedef struct {
    Symbol vars[MAX_VARS];
    int    count;       /* Number of symbols currently stored               */
    int    nextOffset;  /* Next free byte offset in the frame (locals only) */
} SymbolTable;

/* ===========================  LOCAL SCOPE  =============================== */

/* Start a fresh activation record. Called once per function definition. */
void initSymTab(void);

/* Declare a local scalar. Returns its frame offset, or -1 if already declared. */
int addVar(char* name, char* type);

/* Declare a local array of `size` elements. Reserves size*4 bytes in the
 * frame and returns the offset of element 0, or -1 on redeclaration. */
int addArray(char* name, int size);

/* Declare an array PARAMETER. Reserves ONE word to hold the incoming base
 * address (arrays are passed by reference). Returns the offset, or -1. */
int addArrayParam(char* name);

/* ===========================  GLOBAL SCOPE  ============================== */

/* Reset the global table. Called once, before generating any function. */
void initGlobalScope(void);

/* Declare a global scalar / array. Returns 0 on success, -1 on redeclaration. */
int addGlobalVar(char* name, char* type);
int addGlobalArray(char* name, int size);

/* ===========================  LOOKUP  ==================================== */

/* Resolve a name the way the language does: innermost scope first.
 * Returns NULL when the name is not declared anywhere. */
Symbol* lookupSymbol(const char* name);

/* Convenience wrappers kept for readability at the call sites. */
int getVarOffset(char* name);   /* Frame offset, or -1 if global/undeclared  */
int getArraySize(char* name);   /* Element count, or -1 if not an array      */
int isVarDeclared(char* name);  /* 1 if visible in local or global scope     */
int isArray(char* name);        /* 1 if the name denotes an array            */
int isGlobalSymbol(char* name); /* 1 if the name resolves to a global        */

/* Total bytes of frame space handed out so far in the current function.
 * The code generator needs this to size the activation record. */
int getLocalBytes(void);

/* ===========================  TRACING  =================================== */

/* Print the current tables. Used by the -v trace so students can watch
 * storage being allocated as each declaration is processed. */
void printSymTab(void);

#endif
