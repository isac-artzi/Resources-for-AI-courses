# Topic 5 — Compiling Control Flow — Decisions

**Weeks 12–14 · Nov 23 – Dec 13, 2026**

if, else, the dangling-else problem, Boolean operators, and switch.

---

## What you are given

This folder is a **working compiler**. Build it before you change anything:

```bash
cd compiler
make
```

- the complete Topic 4 compiler

## What you are building

- `if` and `if`/`else`, with the dangling-else resolved by precedence
- logical `&&`, `||`, `!` — including the operand normalization
- `switch`, `case`, `default`, fall-through and `break` (optional but recommended)
- the switch semantic checks: one `default`, no duplicate case values

Every place you need to write something is marked `TODO (Topic 5)` in the source,
with the guidance you need at that spot. Work through them in the order they appear
in the file — they are ordered deliberately.

```bash
grep -rn "TODO (Topic 5)" compiler
```

## Building and testing

```bash
cd compiler
make                                   # build ./minicompiler
make test                              # compile and run every tests/*.cm
make clean                             # remove everything make produced

./minicompiler tests/t5_05_nested_switch.cm out.s          # full trace of all six phases
./minicompiler tests/t5_05_nested_switch.cm out.s -q       # quiet: errors and summary only
spim -file out.s                       # run the generated MIPS
```

Compiling also writes `out.tac` and `out.optimized.tac`. Reading those two side by
side is the fastest way to see what the optimizer did.

## You are done when

- all five `t5_*` tests pass, plus everything before them
- `bison -v parser.y` reports **no** conflicts
- `t5_02_dangling_else.cm` prints 2

## The rest of this topic

- **Lecture notes** — `../../docs/topic-5-decisions/lecture-notes.html`
- **Class activities** — `../../docs/topic-5-decisions/` (one per class meeting)
- **The assignment** — `../../docs/topic-5-decisions/Topic-5-Project-5.docx`
- **Grammar reference** — `../../docs/C-Minus-Grammar-Reference.md`

## Requirements

`flex`, `bison`, `gcc`, `make`, and [SPIM](http://spimsimulator.sourceforge.net/)
or QtSPIM to run the generated assembly.
