/* =========================================================================
 *  TOPIC 5 · Compiling Control Flow — Decisions
 * FILE: ast.h   —   Phase 2 — Syntax analysis (the tree it builds)
 * -------------------------------------------------------------------------
 * THE PIPELINE, AND WHERE THIS FILE SITS IN IT
 *   scanner -> parser -> ast -> semantic -> tac -> codegen
 *                        ^^^  this file
 *
 * WHAT IS NEW IN TOPIC 5
 *   • New nodes: IF, SWITCH, CASE
 *
 * WHAT COMES NEXT
 *   Topic 6 adds no new syntax: it measures, documents and hardens what you have.
 *
 * YOUR TASK
 *   Everything below marked  TODO (Topic 5)  is yours to write.
 *   Everything else already works — it is the Topic 4 compiler you
 *   have already built and tested.  Do not rewrite it; extend it.
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
    NODE_FUNC_DEF,    /* Function definition */
    NODE_PARAM,       /* Parameter (name only, type is always int) */
    NODE_PARAM_LIST,  /* Parameter list */
    NODE_FUNC_CALL,   /* Function call */
    NODE_ARG_LIST,    /* Argument list */
    NODE_RETURN,      /* Return statement */
    NODE_WHILE,       /* While loop */
    NODE_FOR,         /* For loop: for (init; condition; update) body */
    NODE_BREAK,       /* Break statement */

    NODE_BLOCK,       /* Block statement { ... } */
    NODE_ARRAY_DECL,  /* Array declaration (e.g., int arr[10] or int arr[]) */
    NODE_ARRAY_INDEX, /* Array indexing (e.g., arr[i]) */
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * Add the decision node kinds:
 *   NODE_IF, NODE_SWITCH, NODE_CASE
 * -------------------------------------------------------------- */
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
            struct ASTNode* arrayLHS;   /* Array index node (NULL for scalar) */
        } assign;
        
        /* Print expression (NODE_PRINT) */
        struct ASTNode* expr;
        
        /* Statement list structure (NODE_STMT_LIST) */
        struct {
            struct ASTNode* stmt;       /* Current statement */
            struct ASTNode* next;       /* Rest of the list */
        } stmtlist;

        /* Function definition (NODE_FUNC_DEF) */
        struct {
            char* name;                 /* Function name */
            struct ASTNode* params;     /* Parameter list */
            struct ASTNode* body;       /* Function body (block or stmt_list) */
        } func_def;

        /* Parameter (NODE_PARAM) - just a name */
        struct {
            char* name;
        } param;

        /* Parameter list (NODE_PARAM_LIST) */
        struct {
            struct ASTNode* param;      /* Current parameter */
            struct ASTNode* next;       /* Rest of parameters */
        } param_list;

        /* Function call (NODE_FUNC_CALL) */
        struct {
            char* name;                 /* Function name */
            struct ASTNode* args;       /* Argument list */
        } func_call;

        /* Argument list (NODE_ARG_LIST) */
        struct {
            struct ASTNode* expr;       /* Current argument expression */
            struct ASTNode* next;       /* Rest of arguments */
        } arg_list;

        /* Return statement (NODE_RETURN) */
        struct {
            struct ASTNode* expr;       /* Expression to return (NULL for void) */
        } ret;

/* --------------------------------------------------------------
 * TODO (Topic 5)
 * A NODE_IF needs three children: condition, then-branch, else-branch.
 * else_stmt is NULL when the source had no `else`.
 * -------------------------------------------------------------- */
        /* While loop (NODE_WHILE) */
        struct {
            struct ASTNode* condition;  /* Loop condition */
            struct ASTNode* body;       /* Loop body */
        } while_stmt;

        /* For loop (NODE_FOR)
         * Represents: for (init; condition; update) body
         * Any part may be NULL:
         *   - init NULL   → no initialization
         *   - condition NULL → always true (infinite until break)
         *   - update NULL → no increment step
         */
        struct {
            struct ASTNode* init;       /* Initialization statement (assignment or NULL) */
            struct ASTNode* condition;  /* Loop condition expression (NULL = always true) */
            struct ASTNode* update;     /* Update statement executed after each iteration */
            struct ASTNode* body;       /* Loop body */
        } for_stmt;

        /* Block statement (NODE_BLOCK) */
        struct {
            struct ASTNode* stmt_list;  /* Statements in block */
        } block;

        /* Array declaration (NODE_ARRAY_DECL) */
        struct {
            char* name;                 /* Array name */
            int size;                   /* Array size (0 for parameters) */
            int isParam;                /* 1 if this is a parameter, 0 otherwise */
        } array_decl;

        /* Array indexing (NODE_ARRAY_INDEX) */
        struct {
            char* name;                 /* Array name */
            struct ASTNode* index;      /* Index expression */
        } array_index;

/* --------------------------------------------------------------
 * TODO (Topic 5)
 * A NODE_SWITCH holds the controlling expression and a LINKED LIST of
 * case clauses, in source order.
 * -------------------------------------------------------------- */
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * A NODE_CASE is one clause of a switch.  It needs: the case constant,
 * a flag saying whether it is the `default` clause, the body statement
 * list (which may be NULL), and a `next` pointer to the following
 * clause.  The `next` pointer is what preserves source order, and
 * source order is what makes fall-through work.
 * -------------------------------------------------------------- */
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
ASTNode* createFuncDef(char* name, ASTNode* params, ASTNode* body);  /* Create function definition */
ASTNode* createParam(char* name);                               /* Create parameter */
ASTNode* createParamList(ASTNode* param, ASTNode* next);        /* Create parameter list */
ASTNode* createFuncCall(char* name, ASTNode* args);             /* Create function call */
ASTNode* createArgList(ASTNode* expr, ASTNode* next);           /* Create argument list */
ASTNode* createReturn(ASTNode* expr);                           /* Create return statement */

/* Control flow nodes */
ASTNode* createBlock(ASTNode* stmt_list);                       /* Create block statement */
ASTNode* createWhile(ASTNode* condition, ASTNode* body);
ASTNode* createFor(ASTNode* init, ASTNode* condition, ASTNode* update, ASTNode* body);
ASTNode* createBreak(void);
/* --------------------------------------------------------------
 * TODO (Topic 5)
 * Declare the constructors for the decision nodes:
 *   createIf(condition, thenStmt, elseStmt)
 *   createSwitch(expr, cases)
 *   createCase(value, isDefault, body)
 * -------------------------------------------------------------- */

/* Array nodes */
ASTNode* createArrayDecl(char* name, int size);                 /* Create array declaration */
ASTNode* createArrayParam(char* name);                          /* Create array parameter */
ASTNode* createArrayIndex(char* name, ASTNode* index);          /* Create array indexing */

/* Text form of a packed operator code (see createBinOp) */
const char* opText(char op);

/* AST DISPLAY FUNCTION */
void printAST(ASTNode* node, int level);                        /* Pretty-print the AST */

#endif