# The Language — Grammar Reference

**Principles of Compiler Design**

This is the language the course compiler accepts, as it grows across the six
topics. Each section shows only what is **new or changed** at that milestone;
everything from earlier topics carries forward unchanged.

Notation is BNF. Terminals appear in `'single quotes'` or in `CAPITALS` when they
come from the scanner as a token with a value; non-terminals are plain lowercase.

> This document is kept in step with the compiler's own grammar. If the two ever
> disagree, `parser.y` is right and this file is a bug — please say so.

---

## Contents

| Topic | Section | What the language gains |
|---|---|---|
| 1 | [Lexical conventions](#lexical-conventions) | the token set |
| 2 | [The starter language](#topic-2--the-starter-language) | `int`, assignment, `+`, `print` |
| 3 | [Arrays and functions](#topic-3--arrays-and-functions) | `- * /`, arrays, functions, scope |
| 4 | [Loops](#topic-4--loops) | relational operators, `while`, `for`, `break` |
| 5 | [Decisions](#topic-5--decisions) | `if`/`else`, `&& \|\| !`, `switch` |
| — | [Complete grammar](#the-complete-grammar) | everything, in one place |
| — | [Precedence table](#operator-precedence) | |
| — | [Differences from C](#differences-from-c) | |
| — | [Reserved identifiers](#reserved-identifiers) | |

---

## Lexical conventions

These apply at every milestone. They are the whole of Topic 1's deliverable.

| Element | Pattern | Examples |
|---|---|---|
| Identifier — `ID` | letter or `_`, then letters, digits or `_` | `x`, `count`, `my_var` |
| Integer literal — `NUM` | one or more digits | `0`, `42`, `100` |
| Line comment | `//` to end of line | `// note` |
| Block comment | `/*` … `*/`, may span lines | `/* note */` |
| Whitespace | space, tab, newline | ignored, but counted |

**Keywords** (reserved at every milestone, even before the grammar uses them):

```
int   print   return   if   else   while   for   switch   case   default   break
```

**Operators and delimiters:**

```
+  -  *  /            arithmetic
<  >  <=  >=  ==  !=  relational       (used from Topic 4)
&&  ||  !             logical          (used from Topic 5)
=                     assignment
;  :  ,               delimiters
(  )  {  }  [  ]      grouping
```

Two rules resolve every ambiguity in the scanner, in this order:

1. **Longest match.** `<=` beats `<` because it consumes more characters. Where the
   rules appear in the file is irrelevant to this.
2. **First match**, on a tie. `int` matches both the keyword rule and the identifier
   rule, at three characters each — so the keyword rule must be written first.

---

## Topic 2 — the starter language

A program is a flat list of statements. There are no functions, no blocks, and one
arithmetic operator.

```
program     ->  stmt_list

stmt_list   ->  stmt
              | stmt_list stmt

stmt        ->  decl
              | assign
              | print_stmt

decl        ->  'int' ID ';'

assign      ->  ID '=' expr ';'

expr        ->  NUM
              | ID
              | expr '+' expr

print_stmt  ->  'print' '(' expr ')' ';'
```

### Example

```c
int x;
int y;
int total;

x = 5;
y = 10;
total = x + y + 3;

print(total);        // 18
```

### Notes

- Addition is **left associative**: `a + b + c` groups as `(a + b) + c`.
- All variables are declared at the top level; there is no scope to speak of yet.
- The code generator wraps the whole statement list in an implicit `main()`, so the
  generated assembly already has the entry point every later milestone needs.

---

## Topic 3 — arrays and functions

The shape of a program changes: it is no longer a list of statements but a list of
**declarations and function definitions**. Executable statements now live only
inside a function body, which is why `main` suddenly matters.

### New and changed rules

```
program            ->  decl_or_func_list

decl_or_func_list  ->  decl_or_func
                     | decl_or_func_list decl_or_func

decl_or_func       ->  func_def
                     | decl

func_def           ->  'int' ID '(' params ')' block
                     | 'int' ID '(' ')' block

params             ->  param_list

param_list         ->  param
                     | param_list ',' param

param              ->  'int' ID
                     | 'int' ID '[' ']'          -- array parameter: no size

block              ->  '{' stmt_list '}'
                     | '{' '}'

stmt               ->  ... (as before)
                     | return_stmt
                     | block
                     | func_call ';'

decl               ->  'int' ID ';'
                     | 'int' ID '[' NUM ']' ';'  -- size must be a literal

assign             ->  ID '=' expr ';'
                     | ID '[' expr ']' '=' expr ';'

return_stmt        ->  'return' expr ';'
                     | 'return' ';'

expr               ->  NUM
                     | ID
                     | ID '[' expr ']'
                     | func_call
                     | expr '+' expr
                     | expr '-' expr
                     | expr '*' expr
                     | expr '/' expr
                     | '-' expr        %prec UMINUS
                     | '(' expr ')'

func_call          ->  ID '(' args ')'
                     | ID '(' ')'

args               ->  (empty)
                     | arg_list

arg_list           ->  expr
                     | arg_list ',' expr
```

### Example

```c
int shared[4];                       // a global array

int sum3(int a[]) {                  // array parameter: passed by reference
    return a[0] + a[1] + a[2];
}

int main() {
    int local[3];
    shared[0] = 10; shared[1] = 20; shared[2] = 30; shared[3] = 40;
    print(sum3(shared) + 40);        // 100
    local[0] = 1; local[1] = 2; local[2] = 3;
    print(sum3(local));              // 6
    return 0;
}
```

### Notes

- **Array size must be a literal.** The compiler needs to know how many bytes to
  reserve, and it needs to know at compile time.
- **Arrays are passed by reference.** The callee receives the base address, one
  word, which is why an array parameter carries no size and why the callee cannot
  know the array's length. Every function taking an array in this language also
  takes a count.
- **At most four arguments per call.** They are passed in `$a0`–`$a3`. A fifth is a
  reported error, not silent corruption.
- **Functions may be called before they are defined.** Semantic analysis runs two
  passes: signatures first, bodies second.
- **Scope**: a name is looked up innermost-first, so parameters and locals shadow
  globals.

---

## Topic 4 — loops

### New rules

```
stmt          ->  ... (as before)
                | while_stmt
                | for_stmt
                | break_stmt

while_stmt    ->  'while' '(' expr ')' stmt

for_stmt      ->  'for' '(' for_init ';' for_cond ';' for_update ')' stmt

for_init      ->  (empty)
                | ID '=' expr
                | ID '[' expr ']' '=' expr

for_cond      ->  (empty)                 -- empty means always true
                | expr

for_update    ->  (empty)
                | ID '=' expr
                | ID '[' expr ']' '=' expr

break_stmt    ->  'break' ';'

expr          ->  ... (as before)
                | expr '<'  expr
                | expr '>'  expr
                | expr '<=' expr
                | expr '>=' expr
                | expr '==' expr
                | expr '!=' expr
```

### Example

```c
int main() {
    int i; int s;
    s = 0;
    for (i = 1; i <= 10; i = i + 1) { s = s + i; }
    print(s);                       // 55

    i = 0;
    while (i < 100) { i = i + 7; break; }
    print(i);                       // 7
    return 0;
}
```

### Notes

- **Both loops test at the top.** A loop whose condition is initially false runs
  zero times.
- **All three parts of a `for` header are optional.** An empty condition means
  always true.
- Note what `for_init` is *not*: it is an assignment **without** a semicolon,
  because the `;` belongs to the for-header rather than to the assignment.
- **A comparison yields 1 or 0.** There is no separate boolean type.
- Relational operators bind **looser** than arithmetic, so `a + 1 < b * 2` groups as
  `(a + 1) < (b * 2)`.
- `break` leaves the innermost enclosing loop. Using it outside any loop is a
  *semantic* error, not a syntax error — no context-free grammar can express the
  rule, which makes it the smallest example of why semantic analysis exists.
- There is no `continue` (see [Differences from C](#differences-from-c)).

---

## Topic 5 — decisions

### New rules

```
stmt           ->  ... (as before)
                 | if_stmt
                 | switch_stmt

if_stmt        ->  'if' '(' expr ')' stmt   %prec LOWER_THAN_ELSE
                 | 'if' '(' expr ')' stmt 'else' stmt

switch_stmt    ->  'switch' '(' expr ')' '{' case_list '}'

case_list      ->  (empty)
                 | case_list case_clause

case_clause    ->  'case' NUM ':' opt_stmt_list
                 | 'default'   ':' opt_stmt_list

opt_stmt_list  ->  (empty)
                 | stmt_list

expr           ->  ... (as before)
                 | expr '&&' expr
                 | expr '||' expr
                 | '!' expr        %prec NOT
```

### Example

```c
int grade(int s) {
    switch (s) {
        case 1: return 10;
        case 2: return 20;
        case 3: return 30;
        default: return 99;
    }
}

int main() {
    int x; x = 5;
    if (x > 0 && x < 10) { print(1); } else { print(0); }

    x = 4;
    switch (x) {
        case 4:                    // empty body: falls through
        case 5: print(45); break;
        default: print(0); break;
    }
    print(grade(3));               // 30
    return 0;
}
```

### Notes

- **The dangling else binds to the nearest `if`.** `if (a) if (b) X else Y` means
  `if (a) { if (b) X else Y }`. The grammar is ambiguous; the ambiguity is resolved
  by giving the else-less rule lower precedence than the `ELSE` token, so the parser
  shifts rather than reduces. Written this way, bison reports **no conflicts** — and
  a grammar with unexplained conflicts is a grammar nobody is reading.
- **Truthiness is C-style**: any non-zero value is true, and `&&`, `||`, `!` all
  produce exactly 0 or 1. Note that MIPS `and` and `or` are *bitwise*, so each
  operand must be normalized to 0/1 before they are applied.
- **`&&` and `||` evaluate both operands** (they do not short-circuit). This is
  observable only if an operand has a side effect — which it can, since expressions
  may contain calls. Short-circuiting is listed as an optional extension in Project 5.
- **Case values must be literals**, and the controlling expression is evaluated
  exactly once.
- **Fall-through is the default.** A case body with no `break` runs into the next
  one. That is not a special mechanism: the case bodies are laid out consecutively
  with no jump between them, so falling through is simply what happens.
- **`break` inside a switch leaves the switch**, not an enclosing loop, because the
  switch pushed its exit label more recently.

---

## The complete grammar

Everything above, in one place — the language as of Topic 5.

```
program            ->  decl_or_func_list

decl_or_func_list  ->  decl_or_func  |  decl_or_func_list decl_or_func
decl_or_func       ->  func_def  |  decl

func_def           ->  'int' ID '(' params ')' block  |  'int' ID '(' ')' block
params             ->  param_list
param_list         ->  param  |  param_list ',' param
param              ->  'int' ID  |  'int' ID '[' ']'

block              ->  '{' stmt_list '}'  |  '{' '}'
stmt_list          ->  stmt  |  stmt_list stmt

stmt               ->  decl | assign | print_stmt | return_stmt
                     | while_stmt | for_stmt | break_stmt
                     | if_stmt | switch_stmt
                     | block | func_call ';'

decl               ->  'int' ID ';'  |  'int' ID '[' NUM ']' ';'
assign             ->  ID '=' expr ';'  |  ID '[' expr ']' '=' expr ';'
print_stmt         ->  'print' '(' expr ')' ';'
return_stmt        ->  'return' expr ';'  |  'return' ';'
break_stmt         ->  'break' ';'

if_stmt            ->  'if' '(' expr ')' stmt  %prec LOWER_THAN_ELSE
                     | 'if' '(' expr ')' stmt 'else' stmt

while_stmt         ->  'while' '(' expr ')' stmt
for_stmt           ->  'for' '(' for_init ';' for_cond ';' for_update ')' stmt
for_init           ->  (empty) | ID '=' expr | ID '[' expr ']' '=' expr
for_cond           ->  (empty) | expr
for_update         ->  (empty) | ID '=' expr | ID '[' expr ']' '=' expr

switch_stmt        ->  'switch' '(' expr ')' '{' case_list '}'
case_list          ->  (empty) | case_list case_clause
case_clause        ->  'case' NUM ':' opt_stmt_list | 'default' ':' opt_stmt_list
opt_stmt_list      ->  (empty) | stmt_list

expr               ->  NUM | ID | ID '[' expr ']' | func_call | '(' expr ')'
                     | expr '+' expr  | expr '-' expr
                     | expr '*' expr  | expr '/' expr
                     | expr '<' expr  | expr '>' expr
                     | expr '<=' expr | expr '>=' expr
                     | expr '==' expr | expr '!=' expr
                     | expr '&&' expr | expr '||' expr
                     | '-' expr  %prec UMINUS
                     | '!' expr  %prec NOT

func_call          ->  ID '(' args ')'  |  ID '(' ')'
args               ->  (empty) | arg_list
arg_list           ->  expr | arg_list ',' expr
```

---

## Operator precedence

Lowest first. Within bison, each successive declaration binds **tighter** than the
one above it, which is why the order of these lines is the whole of the precedence
specification.

| Precedence | Operators | Associativity | Arrives in |
|---|---|---|---|
| 1 (loosest) | `LOWER_THAN_ELSE`, `ELSE` | nonassoc | Topic 5 |
| 2 | `\|\|` | left | Topic 5 |
| 3 | `&&` | left | Topic 5 |
| 4 | `==` `!=` | left | Topic 4 |
| 5 | `<` `>` `<=` `>=` | left | Topic 4 |
| 6 | `+` `-` | left | Topic 2 / 3 |
| 7 | `*` `/` | left | Topic 3 |
| 8 | unary `-` (`UMINUS`) | right | Topic 3 |
| 9 (tightest) | unary `!` (`NOT`) | right | Topic 5 |

Worked examples:

```
2 + 3 * 4          =>  2 + (3 * 4)      =  14
(2 + 3) * 4        =>  20
17 - 5 - 3         =>  (17 - 5) - 3     =  9      (left associative)
-2 * 3             =>  (-2) * 3         = -6      (%prec UMINUS)
a + 1 < b * 2      =>  (a + 1) < (b * 2)
x > 0 && y > 0     =>  (x > 0) && (y > 0)
a || b && c        =>  a || (b && c)               (&& binds tighter)
```

---

## Differences from C

The language is a subset of C, chosen so that the whole compiler stays readable.
Each omission is a deliberate simplification and several of them are listed as
optional extensions in the assignment descriptions.

| Not supported | Why, and where it is discussed |
|---|---|
| any type but `int` | no type system; the type is not tracked past the declaration. Project 3 optional. |
| pointers, `&`, `*` deref | arrays already give indirection where it is needed |
| `for` with a declaration in the init | `for (int i = 0; ...)` — the init is an assignment, not a declaration |
| `i++`, `+=` | assignment is a statement, never an expression, which removes a class of parsing problems |
| `continue` | Project 4 optional; harder than it looks, because in a `for` loop it must jump to the update, not the top |
| `do ... while` | Project 4 optional; two lines different from `while` |
| short-circuit `&&` / `\|\|` | Project 5 optional; ours evaluates both operands |
| the ternary `? :` | Project 5 optional; lowers identically to `if`/`else` |
| `struct` | Project 3 optional; the most substantial extension available |
| string and character literals | Project 1 optional; `print` takes an integer only |
| more than 4 arguments | Project 3 optional; arguments are passed in `$a0`–`$a3` |
| array bounds checking | Project 3 optional; no length survives to run time |
| separate compilation, `#include` | out of scope for the course |

---

## Reserved identifiers

The intermediate-code generator invents names of its own. If a user variable had one
of those shapes, the back end could not tell them apart — so the language reserves
that namespace and the semantic analyzer rejects it with an explanation.

| Reserved | Used for |
|---|---|
| `t0`, `t1`, … | three-address-code temporaries |
| `L0`, `L1`, … | generated labels |
| anything starting `__sw` | the hidden variable holding a `switch` controlling expression |

```
int t0;      // error: 't0' is reserved for the compiler's own use
int total;   // fine
```

In the generated assembly, user names are decorated so that they cannot collide with
MIPS instruction mnemonics: a function `add` becomes the label `fn_add`, and a global
`counter` becomes `g_counter`. Without that, a user function named `add` would emit
the label `add:` and the assembler would reject a file the programmer never wrote.
