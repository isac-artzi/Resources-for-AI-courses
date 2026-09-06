#!/usr/bin/env python3
"""Assemble the GitHub Pages site from each course's tutorials folder.

Every top-level directory holding a "Tutorials/" (or "tutorials/") folder with HTML
in it is published to the site as <course-slug>/. A per-course index is generated
unless the folder already ships its own index.html, and a landing page linking every
course is generated at the site root. Nothing in the repository is moved or
duplicated: this runs at build time inside the Pages workflow.

Usage:  python3 .github/build-site.py [output-dir]     # default: _site
"""

import html
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "_site"
SITE_TITLE = "Resources for AI Courses"
AUTHOR = "Isac Artzi"
KINDS = [("exercise", "Exercises"), ("lecture_notes", "Lecture notes"), ("tutorial", "Tutorial")]


def slug(name):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


def title_of(path):
    head = path.read_text(encoding="utf-8", errors="replace")[:8000]
    m = re.search(r"<title>(.*?)</title>", head, re.S | re.I)
    return html.unescape(re.sub(r"\s+", " ", m.group(1))).strip() if m else path.stem


def subject_of(title):
    """'Topic 3 Tutorial - Convolutional Neural Networks' -> 'Convolutional Neural Networks'."""
    s = re.sub(r"^\s*Topic\s*\d+\s*", "", title)
    s = re.sub(r"^[\s·—–-]+", "", s)
    parts = [p.strip() for p in re.split(r"\s[·—–-]\s", s) if p.strip()]
    parts = [p for p in parts if p.lower() not in
             ("tutorial", "tutorials", "exercises", "exercise", "lecture notes")]
    return max(parts, key=len) if parts else s


def kind_of(path):
    n = path.name.lower()
    for needle, label in KINDS:
        if needle in n:
            return label
    return "Page"


def topic_of(path, title):
    m = re.search(r"[Tt]opic[ _-]?(\d+)", title) or re.search(r"[Tt]opic[ _-]?(\d+)", str(path))
    return int(m.group(1)) if m else 999


def tutorials_dir(course):
    for name in ("Tutorials", "tutorials"):
        d = course / name
        if d.is_dir() and any(d.rglob("*.html")):
            return d
    return None


def collect():
    courses = []
    for course in sorted(p for p in ROOT.iterdir()
                         if p.is_dir() and not p.name.startswith((".", "_"))):
        src = tutorials_dir(course)
        if src is None:
            continue
        course_slug = slug(course.name)
        dest = OUT / course_slug
        pages = []
        for f in sorted(src.rglob("*.html")):
            rel = f.relative_to(src).as_posix()
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            if f.name.lower() != "index.html":
                t = title_of(f)
                pages.append({"n": topic_of(f, t), "kind": kind_of(f),
                              "subject": subject_of(t), "href": rel})
        topics = {}
        for p in pages:
            topics.setdefault(p["n"], []).append(p)
        courses.append({"name": course.name, "slug": course_slug,
                        "topics": dict(sorted(topics.items())),
                        "own_index": (src / "index.html").is_file(),
                        "count": len(pages)})
        print(f"  {course.name} -> {course_slug}/ "
              f"({len(topics)} topics, {len(pages)} pages"
              f"{', own index kept' if (src / 'index.html').is_file() else ''})")
    return courses


CSS = """
:root{--bg:#fbfbfd;--paper:#fff;--ink:#1d2233;--ink-2:#4a5169;--ink-3:#7b819a;
--line:#e3e6ef;--accent:#3b4fd8;--accent-2:#2a3aa6;--accent-soft:#eef0fd;--radius:10px;
--shadow:0 1px 2px rgba(20,25,50,.06),0 6px 20px rgba(20,25,50,.05);
--font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Helvetica,Arial,sans-serif;
color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);
font-size:16.5px;line-height:1.6;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.site-header{background:var(--paper);border-bottom:1px solid var(--line)}
.inner{max-width:940px;margin:0 auto;padding:1.4rem 1.25rem}
.site-header h1{font-size:1.5rem;margin:0;letter-spacing:-.01em}
.site-header p{margin:.45rem 0 0;color:var(--ink-2);font-size:.97rem}
.crumbs{font-size:.82rem;color:var(--ink-3);margin:0 0 .3rem}
main.inner{padding-top:1.8rem;padding-bottom:3rem}
.course{background:var(--paper);border:1px solid var(--line);border-radius:var(--radius);
box-shadow:var(--shadow);padding:1.4rem 1.5rem;margin-bottom:1.5rem}
.course h2{font-size:1.25rem;margin:0 0 .2rem;letter-spacing:-.01em}
.meta{font-size:.82rem;color:var(--ink-3);margin:0 0 1rem}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th,td{text-align:left;padding:.55rem .7rem;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--accent-soft);color:var(--accent-2);font-weight:650}
td.num{color:var(--ink-3);width:3.5rem;font-variant-numeric:tabular-nums}
.open{display:inline-block;background:var(--accent-soft);color:var(--accent-2);
font-weight:600;font-size:.92rem;padding:.4rem .9rem;border-radius:999px;margin-top:1.1rem}
.open:hover{text-decoration:none;background:#e2e6fc}
.note{color:var(--ink-2);font-size:.92rem;margin:0 0 1.6rem}
footer{border-top:1px solid var(--line);background:var(--paper)}
footer .inner{display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;
font-size:.85rem;color:var(--ink-3);padding:1rem 1.25rem}
"""


def shell(title, crumbs, body):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header class="site-header"><div class="inner">
{crumbs}  <h1>{html.escape(title)}</h1>
</div></header>
<main class="inner">
{body}
</main>
<footer><div class="inner">
  <span>Created by {html.escape(AUTHOR)}</span><span>{html.escape(SITE_TITLE)}</span>
</div></footer>
</body>
</html>
"""


def course_index(course):
    e = html.escape
    rows = []
    for n, pages in course["topics"].items():
        pages = sorted(pages, key=lambda p: p["kind"] != "Tutorial")
        subject = max((p["subject"] for p in pages), key=len)
        links = " · ".join(f'<a href="{e(p["href"])}">{e(p["kind"])}</a>' for p in pages)
        rows.append(f'    <tr><td class="num">{n}</td><td>{e(subject)}</td><td>{links}</td></tr>')
    body = (f'  <p class="note">{course["count"]} pages. Each opens in the browser — nothing to '
            "install.</p>\n  <table>\n"
            "    <tr><th>Topic</th><th>Subject</th><th>Pages</th></tr>\n"
            + "\n".join(rows) + "\n  </table>")
    crumbs = '  <p class="crumbs"><a href="../">Resources for AI Courses</a></p>\n'
    return shell(course["name"], crumbs, body)


def landing(courses):
    e = html.escape
    blocks = []
    for c in courses:
        rows = []
        for n, pages in c["topics"].items():
            subject = max((p["subject"] for p in pages), key=len)
            rows.append(f"      <li>{e(subject)}</li>")
        blocks.append(
            f"""    <section class="course">
      <h2><a href="{e(c['slug'])}/">{e(c['name'])}</a></h2>
      <p class="meta">{len(c['topics'])} topics · {c['count']} pages</p>
      <ol>
{chr(10).join(rows)}
      </ol>
      <a class="open" href="{e(c['slug'])}/">Open the tutorials →</a>
    </section>""")
    body = ('  <p class="note">Tutorials, lecture notes and exercises for each course. Project '
            'templates and starter code live in the '
            '<a href="https://github.com/isac-artzi/Resources-for-AI-courses">repository</a>.</p>\n'
            + "\n".join(blocks))
    return shell(SITE_TITLE, "", body)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    print(f"Building site into {OUT}")
    courses = collect()
    if not courses:
        sys.exit("No course folders with a tutorials folder were found")
    for c in courses:
        if not c["own_index"]:
            (OUT / c["slug"] / "index.html").write_text(course_index(c), encoding="utf-8")
    (OUT / "index.html").write_text(landing(courses), encoding="utf-8")
    print(f"Landing page written · {len(courses)} course(s)")


if __name__ == "__main__":
    main()
