# Principles of Compiler Design

## ▶ [Open the course site](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/)

Lecture notes, in-class activities, project descriptions and the language reference all live on the
site. This folder also holds the starter code you build on.

The site files are HTML. **Use the link above** — github.com displays `.html` as source code, so
clicking one in the file browser shows markup instead of the page.

## The six compilers

The course does not build one compiler in six pieces. It builds **six compilers**, each the previous
one with a new language feature threaded through all six phases — scanner, parser, semantic
analysis, IR, optimization, code generation.

| Topic | Site | Starter code |
|---|---|---|
| 1 · Compiler Design Phases | [Lecture notes and activities](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/topic-1-lexical-analysis/) | [`student/topic-1-lexical-analysis`](./student/topic-1-lexical-analysis) |
| 2 · Compiler for a Starter Language | [Lecture notes and activities](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/topic-2-minimal-compiler/) | [`student/topic-2-minimal-compiler`](./student/topic-2-minimal-compiler) |
| 3 · Compiling Complex Variables and Functions | [Lecture notes and activities](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/topic-3-arrays-and-functions/) | [`student/topic-3-arrays-and-functions`](./student/topic-3-arrays-and-functions) |
| 4 · Compiling Loops | [Lecture notes and activities](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/topic-4-loops/) | [`student/topic-4-loops`](./student/topic-4-loops) |
| 5 · Compiling Control Flow — Decisions | [Lecture notes and activities](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/topic-5-decisions/) | [`student/topic-5-decisions`](./student/topic-5-decisions) |
| 6 · Compiler Design and Implementation | [Lecture notes and activities](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/topic-6-complete-compiler/) | [`student/topic-6-complete-compiler`](./student/topic-6-complete-compiler) |

Also on the site: [the toolchain](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/toolchain.html) (flex, bison,
make, and how they fit together) and the
[C-Minus grammar reference](https://isac-artzi.github.io/Resources-for-AI-courses/principles-of-compiler-design/C-Minus-Grammar-Reference.md).

## In this folder

- [`docs/`](./docs) — the course site: lecture notes, activities, project documents, assets
- [`student/`](./student) — starter code for all six topics, one folder per topic, each with its
  own `README.md`, `Makefile` and test inputs

Requires flex, bison, gcc and make. Each topic folder builds with `make`.

Created by Isac Artzi.
