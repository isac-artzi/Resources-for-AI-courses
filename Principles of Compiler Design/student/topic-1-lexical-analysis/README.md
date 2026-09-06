# Topic 1 — Compiler Design Phases

**Week 1 · Sep 8 – Sep 13, 2026**

What the six phases are, and the first one built for real.

---

## What you are given

This folder is a **working compiler**. Build it before you change anything:

```bash
cd lexer
make
```

- `tokens.h` — the token kinds, and the counters the scanner maintains
- `main.c` — the driver that prints the token table and the summary
- `Makefile`, and six test files under `tests/`

## What you are building

- `scanner.l` — every rule, in the order the TODO comments give
- `tokenName()` and `tokenCategory()` in `main.c`

Every place you need to write something is marked `TODO (Topic 1)` in the source,
with the guidance you need at that spot. Work through them in the order they appear
in the file — they are ordered deliberately.

```bash
grep -rn "TODO (Topic 1)" lexer
```

## Building and testing

```bash
cd lexer
make                                   # build ./lexer
make test                              # run the lexer over every tests/*.cm
make clean                             # remove everything make produced

./lexer tests/01_all_tokens.cm                # the full token table
./lexer tests/01_all_tokens.cm -c             # counts and summary only
echo $?                                # 0 = lexically clean, 1 = errors found
```

The exit status matters: it is what lets a script tell success from failure without
reading the output.

## You are done when

- `make test` reports 0 errors for tests 01–04
- `./lexer tests/05_lexical_errors.cm` reports **four** errors and exits 1
- `./lexer tests/06_unterminated_comment.cm` reports one error and exits 1

## The rest of this topic

- **Lecture notes** — `../../docs/topic-1-lexical-analysis/lecture-notes.html`
- **Class activities** — `../../docs/topic-1-lexical-analysis/` (one per class meeting)
- **The assignment** — `../../docs/topic-1-lexical-analysis/Topic-1-Project-1.docx`
- **Grammar reference** — `../../docs/C-Minus-Grammar-Reference.md`

## Requirements

`flex`, `bison`, `gcc`, `make`, and [SPIM](http://spimsimulator.sourceforge.net/)
or QtSPIM to run the generated assembly.
