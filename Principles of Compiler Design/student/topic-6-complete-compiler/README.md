# Topic 6 — Compiler Design and Implementation

**Week 15 · Dec 14 – Dec 20, 2026**

Measure it, document it, defend it.

---

## What you are given

This folder is a **working compiler**. Build it before you change anything:

```bash
cd compiler
make
```

- the complete compiler, and the full cumulative test suite

## What you are building

- nothing new in the language
- a README a stranger can build from
- compilation-time and execution-time measurements
- an honest limitations section

Every place you need to write something is marked `TODO (Topic 6)` in the source,
with the guidance you need at that spot. Work through them in the order they appear
in the file — they are ordered deliberately.

```bash
grep -rn "TODO (Topic 6)" compiler
```

## Building and testing

```bash
cd compiler
make                                   # build ./minicompiler
make test                              # compile and run every tests/*.cm
make clean                             # remove everything make produced

./minicompiler tests/t6_07_benchmark.cm out.s          # full trace of all six phases
./minicompiler tests/t6_07_benchmark.cm out.s -q       # quiet: errors and summary only
spim -file out.s                       # run the generated MIPS
```

Compiling also writes `out.tac` and `out.optimized.tac`. Reading those two side by
side is the fastest way to see what the optimizer did.

## You are done when

- every test from Topics 2–5 passes
- `t6_06_everything.cm` runs — that program exercises every feature at once
- someone outside the team has built it from the README alone

## The rest of this topic

- **Lecture notes** — `../../docs/topic-6-complete-compiler/lecture-notes.html`
- **Class activities** — `../../docs/topic-6-complete-compiler/` (one per class meeting)
- **The assignment** — `../../docs/topic-6-complete-compiler/Topic-6-Project-6.docx`
- **Grammar reference** — `../../docs/C-Minus-Grammar-Reference.md`

## Requirements

`flex`, `bison`, `gcc`, `make`, and [SPIM](http://spimsimulator.sourceforge.net/)
or QtSPIM to run the generated assembly.
