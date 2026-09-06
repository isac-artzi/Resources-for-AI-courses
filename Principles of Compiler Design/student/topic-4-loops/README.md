# Topic 4 — Compiling Loops

**Weeks 9–11 · Nov 2 – Nov 22, 2026**

Control flow becomes labels and jumps; then the optimizer earns its keep.

---

## What you are given

This folder is a **working compiler**. Build it before you change anything:

```bash
cd compiler
make
```

- the complete Topic 3 compiler

## What you are building

- relational operators, and their place in the precedence table
- `while` and `for`, lowered to labels and jumps
- `break`, with the label stack and the depth check
- label and branch emission in the back end
- branch simplification in the optimizer

Every place you need to write something is marked `TODO (Topic 4)` in the source,
with the guidance you need at that spot. Work through them in the order they appear
in the file — they are ordered deliberately.

```bash
grep -rn "TODO (Topic 4)" compiler
```

## Building and testing

```bash
cd compiler
make                                   # build ./minicompiler
make test                              # compile and run every tests/*.cm
make clean                             # remove everything make produced

./minicompiler tests/t4_06_optimize.cm out.s          # full trace of all six phases
./minicompiler tests/t4_06_optimize.cm out.s -q       # quiet: errors and summary only
spim -file out.s                       # run the generated MIPS
```

Compiling also writes `out.tac` and `out.optimized.tac`. Reading those two side by
side is the fastest way to see what the optimizer did.

## You are done when

- all six `t4_*` tests pass, plus everything from Topics 2 and 3
- the optimizer reports a scorecard and reaches a fixed point
- a benchmark has been measured optimized vs unoptimized

## The rest of this topic

- **Lecture notes** — `../../docs/topic-4-loops/lecture-notes.html`
- **Class activities** — `../../docs/topic-4-loops/` (one per class meeting)
- **The assignment** — `../../docs/topic-4-loops/Topic-4-Project-4.docx`
- **Grammar reference** — `../../docs/C-Minus-Grammar-Reference.md`

## Requirements

`flex`, `bison`, `gcc`, `make`, and [SPIM](http://spimsimulator.sourceforge.net/)
or QtSPIM to run the generated assembly.
