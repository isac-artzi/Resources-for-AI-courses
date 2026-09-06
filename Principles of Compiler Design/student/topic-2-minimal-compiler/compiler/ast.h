/* =========================================================================
 *  TOPIC 2 · Compiler for a Starter Language
 * FILE: ast.h   —   Phase 2 — Syntax analysis (the tree it builds)
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                        ^^^  this file
 *
 * WHAT IS NEW IN TOPIC 2
 *   • The node kinds the starter language needs: NUM, VAR, BINOP,
 *   • DECL, ASSIGN, PRINT, STMT_LIST
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

#ifndef AST_H
#define AST_H

/* ABSTRACT SYNTAX TREE (AST)
 * The AST is an intermediate representation of the program structure
 * It represents the hierarchical syntax of the source code
 * Each node represents a construct in the language
 */

/* NODE TYPES - Different kinds of AST nodes in our language */
typedef enum {
    NODE_NUM,         /* Numeric literal (e.g., 42) */
    NODE_VAR,         /* Variable reference (e.g., x) */
    NODE_BINOP,       /* Binary operation (e.g., x + y) */
    NODE_DECL,        /* Variable declaration (e.g., int x) */
    NODE_ASSIGN,      /* Assignment statement (e.g., x = 10) */
    NODE_PRINT,       /* Print statement (e.g., print(x)) */
    NODE_STMT_LIST,   /* List of statements (program structure) */
} NodeType;

/* AST NODE STRUCTURE
 * Uses a union to efficiently store different node data
 * Only the relevant fields for each node type are used
 */
typedef struct ASTNode {
    NodeType type;  /* Identifies what kind of node this is */
    int lineno;     /* Line number in source code for error reporting */

    /* Union allows same memory to store different data types */
    union {
        /* Literal number value (NODE_NUM) */
        int num;
        
        /* Variable reference name (NODE_VAR) */
        char* name;

        /* Declaration structure (NODE_DECL) */
        struct {
            char* name;
            char* varType;
        } decl;


        /* Binary operation structure (NODE_BINOP) */
        struct {
            char op;                    /* Operator character ('+') */
            struct ASTNode* left;       /* Left operand */
            struct ASTNode* right;      /* Right operand */
        } binop;
        
        /* Assignment structure (NODE_ASSIGN) */
        struct {
            char* var;                  /* Variable being assigned to (NULL for array) */
            struct ASTNode* value;      /* Expression being assigned */
        } assign;
        
        /* Print expression (NODE_PRINT) */
        struct ASTNode* expr;
        
        /* Statement list structure (NODE_STMT_LIST) */
        struct {
            struct ASTNode* stmt;       /* Current statement */
            struct ASTNode* next;       /* Rest of the list */
        } stmtlist;

    } data;
} ASTNode;

/* AST CONSTRUCTION FUNCTIONS
 * These functions are called by the parser to build the tree
 */
/* Basic nodes */
ASTNode* createNum(int value);                                   /* Create number node */
ASTNode* createVar(char* name);                                  /* Create variable node */
ASTNode* createBinOp(char op, ASTNode* left, ASTNode* right);   /* Create binary op node */
ASTNode* createDecl(char* type, char* name);                     /* Create declaration node */
ASTNode* createAssign(char* var, ASTNode* value);               /* Create assignment node */
ASTNode* createPrint(ASTNode* expr);                            /* Create print node */
ASTNode* createStmtList(ASTNode* stmt1, ASTNode* stmt2);        /* Create statement list */

/* Function-related nodes */

/* Control flow nodes */


/* Text form of a packed operator code (see createBinOp) */
const char* opText(char op);

/* AST DISPLAY FUNCTION */
void printAST(ASTNode* node, int level);                        /* Pretty-print the AST */

#endif