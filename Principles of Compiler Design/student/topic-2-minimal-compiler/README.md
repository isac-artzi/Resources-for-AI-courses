# Topic 2 — Compiler for a Starter Language

**Weeks 2–5 · Sep 14 – Oct 11, 2026**

A whole compiler, end to end, for the smallest language worth compiling.

---

## What you are given

This folder is a **working compiler**. Build it before you change anything:

```bash
cd compiler
make
```

- `scanner.l` — a reference version of the Project 1 scanner, adapted to return bison token numbers
- every `.h` file — the data structures are the design
- `parser.y` prologue: `%union`, `%token`, the precedence table, `yyerror`
- `tac.c` machinery: temporary and label allocation, list handling
- `codegen.c` parts 1–3: the register cache, addressing, frame layout
- `main.c` — the six-phase driver

## What you are building

- the grammar rules and their semantic actions (`parser.y`)
- the AST constructors and `printAST` (`ast.c`)
- symbol table insert and lookup (`symtab.c`)
- the semantic checks (`semantic.c`)
- AST → TAC (`tac.c`)
- one optimization pass (`tac.c`)
- TAC → MIPS instruction selection (`codegen.c`)

Every place you need to write something is marked `TODO (Topic 2)` in the source,
with the guidance you need at that spot. Work through them in the order they appear
in the file — they are ordered deliberately.

```bash
grep -rn "TODO (Topic 2)" compiler
```

## Building and testing

```bash
cd compiler
make                                   # build ./minicompiler
make test                              # compile and run every tests/*.cm
make clean                             # remove everything make produced

./minicompiler tests/t2_05_errors_syntax.cm out.s          # full trace of all six phases
./minicompiler tests/t2_05_errors_syntax.cm out.s -q       # quiet: errors and summary only
spim -file out.s                       # run the generated MIPS
```

Compiling also writes `out.tac` and `out.optimized.tac`. Reading those two side by
side is the fastest way to see what the optimizer did.

## You are done when

- `make test` runs every program in `tests/`
- `t2_01`–`t2_03` produce the outputs in their header comments
- `t2_04` and `t2_05` are supposed to FAIL — check the messages name the right line

## The rest of this topic

- **Lecture notes** — `../../docs/topic-2-minimal-compiler/lecture-notes.html`
- **Class activities** — `../../docs/topic-2-minimal-compiler/` (one per class meeting)
- **The assignment** — `../../docs/topic-2-minimal-compiler/Topic-2-Project-2.docx`
- **Grammar reference** — `../../docs/C-Minus-Grammar-Reference.md`

## Requirements

`flex`, `bison`, `gcc`, `make`, and [SPIM](http://spimsimulator.sourceforge.net/)
or QtSPIM to run the generated assembly.
