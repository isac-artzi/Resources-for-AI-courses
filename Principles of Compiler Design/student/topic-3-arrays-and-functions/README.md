# Topic 3 — Compiling Complex Variables and Functions

**Weeks 6–8 · Oct 12 – Nov 1, 2026**

Arrays, functions, scope — and the activation record that makes calls possible.

---

## What you are given

This folder is a **working compiler**. Build it before you change anything:

```bash
cd compiler
make
```

- the complete, working Topic 2 compiler — it builds and passes the Topic 2 tests before anything is written

## What you are building

- arithmetic: `- * /`, parentheses, unary minus, and the precedence table
- arrays: declaration, indexing, array parameters
- functions: definitions, parameters, calls, `return`
- scope: a scope stack, and two-pass semantic analysis
- activation records: prologue, epilogue, saved `$ra`, arguments in `$a0`–`$a3`

Every place you need to write something is marked `TODO (Topic 3)` in the source,
with the guidance you need at that spot. Work through them in the order they appear
in the file — they are ordered deliberately.

```bash
grep -rn "TODO (Topic 3)" compiler
```

## Building and testing

```bash
cd compiler
make                                   # build ./minicompiler
make test                              # compile and run every tests/*.cm
make clean                             # remove everything make produced

./minicompiler tests/t3_05_scope.cm out.s          # full trace of all six phases
./minicompiler tests/t3_05_scope.cm out.s -q       # quiet: errors and summary only
spim -file out.s                       # run the generated MIPS
```

Compiling also writes `out.tac` and `out.optimized.tac`. Reading those two side by
side is the fastest way to see what the optimizer did.

## You are done when

- all five `t3_*` tests produce the outputs in their header comments
- the Topic 2 tests still pass
- `fact(5)` returns 120 — if it does not, the calling convention is wrong

## The rest of this topic

- **Lecture notes** — `../../docs/topic-3-arrays-and-functions/lecture-notes.html`
- **Class activities** — `../../docs/topic-3-arrays-and-functions/` (one per class meeting)
- **The assignment** — `../../docs/topic-3-arrays-and-functions/Topic-3-Project-3.docx`
- **Grammar reference** — `../../docs/C-Minus-Grammar-Reference.md`

## Requirements

`flex`, `bison`, `gcc`, `make`, and [SPIM](http://spimsimulator.sourceforge.net/)
or QtSPIM to run the generated assembly.
