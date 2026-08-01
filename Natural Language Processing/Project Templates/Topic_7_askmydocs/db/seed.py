"""Apply the migration and insert a few demo rows.

    python db/seed.py

Supabase's client library cannot execute arbitrary DDL, so this script does NOT
create the tables for you — it checks whether they exist and tells you what to do
if they don't. Paste db/migrations/001_init.sql into the SQL Editor once; after
that this script is just a way to put rows in the Retrieval Audit tab so you can
see it render before your NLP code works.

WHAT IT DOES NOT DO: it does not write embeddings, because producing one means
loading the embedding model and that is your job in api/nlp.py. The seeded chunks
therefore have a NULL embedding and match_chunks() will never return them — it
filters `embedding is not null` for exactly this reason. Seed data that could be
retrieved would be worse than useless: it would show up in your Ask tab, get
cited, and look like a real result.
"""
from __future__ import annotations

import hashlib
import sys

from api import db

DEMO_DOC = {
    "title": "Demo policy note (seed data)",
    "source": "db/seed.py — synthetic, not part of your real collection",
    "corpus": "retrieval",
}

DEMO_CHUNKS = [
    "The support rota runs Monday to Thursday. Requests received after 16:00 are "
    "queued for the following working day, and the queue is cleared before any "
    "new request is accepted.",
    "Escalation requires two named approvers. The second approver may not be the "
    "person who raised the request, and the approval is recorded against the "
    "request id rather than against a person.",
]


def main() -> int:
    if not db.configured():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set. Copy .env.example "
              "to .env and fill them in.")
        return 1

    if not db.ping():
        print(
            "Could not read the 'documents' table.\n"
            "Open your Supabase project -> SQL Editor -> New query, paste the\n"
            "contents of db/migrations/001_init.sql, and click Run.\n"
            "\n"
            "If it failed on 'type vector does not exist', the pgvector extension\n"
            "is not enabled: the `create extension if not exists vector;` line at\n"
            "the top of the migration has to run before anything else in it.\n"
            "Then re-run me."
        )
        return 1

    text = "\n\n".join(DEMO_CHUNKS)
    doc = db.insert_document(
        title=DEMO_DOC["title"],
        source=DEMO_DOC["source"],
        corpus=DEMO_DOC["corpus"],
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        token_count=sum(len(c.split()) for c in DEMO_CHUNKS),
        doc_metadata={"seed": True},
    )
    if not doc:
        print(
            "insert_document returned None. Either the seed document is already "
            "there (the unique index on content_sha256 refuses a second copy — "
            "that is the corpus-separation guard doing its job) or the write "
            "failed. Check the Supabase logs."
        )
        return 1
    print("inserted document", doc["id"])

    chunk_rows = db.insert_chunks(
        [
            {
                "document_id": doc["id"],
                "ordinal": i,
                "content": c,
                "token_count": len(c.split()),
                "start_token": i * 340,
                "end_token": i * 340 + len(c.split()),
                # Inside the assignment's 300-500 range and 15 percent overlap,
                # so these rows are a worked example of what a real chunk row
                # looks like even though the text itself is short.
                "chunk_size_tokens": 400,
                "overlap_tokens": 60,
                "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding": None,
                "embedding_model": None,
                "embedding_dim": None,
            }
            for i, c in enumerate(DEMO_CHUNKS)
        ]
    )
    print("inserted", len(chunk_rows), "chunks")

    # A fake question, so the Retrieval Audit tab has a row to draw. The question
    # text is not stored anywhere — only its hash, which is the rule the whole
    # logging layer follows.
    query = db.insert_query(
        query_sha256=hashlib.sha256(b"when are late requests handled?").hexdigest(),
        k=len(chunk_rows) or 2,
        model_version="seed",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    if query and chunk_rows:
        class _Fake:
            def __init__(self, chunk_id, similarity, rank):
                self.chunk_id = chunk_id
                self.similarity = similarity
                self.rank = rank

        db.insert_retrievals(
            query["id"],
            [
                _Fake(chunk_rows[0]["id"], 0.81, 1),
                _Fake(chunk_rows[1]["id"], 0.42, 2),
            ][: len(chunk_rows)],
        )
        db.insert_answer(
            query_id=query["id"],
            answer="Requests received after 16:00 are queued for the next working day [1].",
            retrieval_used=True,
            cited_chunk_ids=[chunk_rows[0]["id"]],
            generator_model="seed",
        )
        print("inserted query", query["id"], "with retrievals and an answer")

    print("Done. Open the Retrieval Audit tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
