"""Load the Conditions of Carriage (English and Arabic PDFs) into policy.clauses, linked by clause id.

    python scripts/ingest_policy.py                 # both languages, embedded with bge-m3 when an embedding provider is configured
    python scripts/ingest_policy.py --no-embed      # text only (search_policy reports itself unavailable until embedded)
    python scripts/ingest_policy.py --embed         # require vectors (fails if no embedding provider is configured)

Source: Emirates' public Conditions of Carriage for Passengers and Baggage (data/policy/coc_en.pdf, coc_ar.pdf),
used with credit for demonstration; in the agent's world the document is Falcon's. Parsing and the Arabic
ligature repair live in falcon.policy.parse_coc. Long clauses are split into parts ("7.1#2") of ~1,200 characters
at sentence boundaries; parts share the clause id so citations stay at clause level. Article 1 definitions get
ids 1.1.n and feed policy.glossary (the document's own EN/AR term pairs, on top of Falcon's seeded terms).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from falcon.core.clock import now  # noqa: E402
from falcon.core.config import REPO_ROOT, settings  # noqa: E402
from falcon.policy.parse_coc import parse_both  # noqa: E402

DATA = REPO_ROOT / "data" / "policy"
MAX_PART = 1200
VERSION = date(2024, 12, 28)
DOCS = {
    "en": ("coc-en", "Conditions of Carriage for Passengers and Baggage (Emirates, public document, used with credit)",
           "https://www.emirates.com/english/help/conditions-of-carriage/"),
    "ar": ("coc-ar", "شروط النقل للمسافرين والأمتعة (طيران الإمارات، وثيقة عامة، مع الإشارة للمصدر)",
           "https://www.emirates.com/ae/arabic/help/conditions-of-carriage/"),
}
EXTRACTION = {"en": "pdf-text", "ar": "pdf-text, ligatures repaired from glyph geometry"}


def split_parts(text: str) -> list[str]:
    if len(text) <= MAX_PART:
        return [text]
    sentences = re.split(r"(?<=[.;:؛؟!?])\s+", text)
    parts, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 > MAX_PART and buf:
            parts.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        parts.append(buf.strip())
    return parts


HEADING_MAX = 80


@dataclass
class Row:
    doc_id: str
    clause_id: str          # "7.1", or "7.1#2" for the second part of a long clause
    article_no: int
    article_title: str
    lang: str
    text: str
    section_title: str | None = None   # the short heading row this clause sits under ("Animals"), prepended for embedding
    title_only: bool = False           # a bare heading ("5.5 Aircraft"): kept for get_clause, never embedded


def rows_for(doc_id: str, lang: str, clauses) -> list[Row]:
    rows = []
    for c in clauses:
        for n, part in enumerate(split_parts(c.text), start=1):
            rows.append(Row(doc_id, c.clause_id if n == 1 else f"{c.clause_id}#{n}", c.article_no, c.article_title, lang, part))
    mark_sections(rows)
    return rows


def mark_sections(rows: list[Row]) -> None:
    """Title-only rows ("5.5 Aircraft") held 13% of the index and took result slots without holding an answer; they stay
    readable by clause id but get no vector, and their title is prepended to the clauses under them instead
    ("contextual chunk headers": "Animals: The animal must be properly crated..."). FAILURE_DIARY #34."""
    ids = {r.clause_id for r in rows}
    headings = {r.clause_id: r.text for r in rows if "#" not in r.clause_id and len(r.text) < HEADING_MAX
                and any(other.startswith(r.clause_id + ".") for other in ids)}
    for r in rows:
        if r.clause_id in headings:
            r.title_only = True
        base = r.clause_id.split("#")[0]
        parent = base.rsplit(".", 1)[0] if base.count(".") >= 2 else None
        if parent in headings:
            r.section_title = headings[parent]


def embedding_text(r: Row) -> str:
    return f"{r.article_title} > {r.section_title}: {r.text}" if r.section_title else f"{r.article_title}: {r.text}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="require dense vectors (error if no embedding provider)")
    ap.add_argument("--no-embed", action="store_true", help="load the text only (policy search will report itself unavailable)")
    args = ap.parse_args()

    en, ar, glossary = parse_both(DATA / "coc_en.pdf", DATA / "coc_ar.pdf")
    rows = rows_for(DOCS["en"][0], "en", en) + rows_for(DOCS["ar"][0], "ar", ar)
    with psycopg.connect(settings.dsn_admin, autocommit=False) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM policy.clauses")
        cur.execute("DELETE FROM policy.documents")
        for lang, (doc_id, title, url) in DOCS.items():
            cur.execute("INSERT INTO policy.documents(id,lang,title,source_url,version_date,extraction,loaded_at) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (doc_id, lang, title, url, VERSION, EXTRACTION[lang], now()))
        cur.executemany(
            "INSERT INTO policy.clauses(doc_id,clause_id,article_no,article_title,lang,text,char_len) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            [(r.doc_id, r.clause_id, r.article_no, r.article_title, r.lang, r.text, len(r.text)) for r in rows], returning=True)
        ids = []
        while True:
            ids.append(cur.fetchone()[0])
            if not cur.nextset():
                break
        cur.executemany(
            "INSERT INTO policy.glossary(term_en,term_ar,note) VALUES (%s,%s,%s) ON CONFLICT (term_en) DO UPDATE SET term_ar = EXCLUDED.term_ar, note = EXCLUDED.note",
            [(en_t, ar_t, "Article 1 definition") for en_t, ar_t in glossary if ar_t])
        conn.commit()
        en_ids = {c.clause_id for c in en}
        ar_ids = {c.clause_id for c in ar}
        n_en, n_ar = sum(r.lang == "en" for r in rows), sum(r.lang == "ar" for r in rows)
        print(f"loaded {n_en} EN rows ({len(en)} clauses), {n_ar} AR rows ({len(ar)} clauses); "
              f"clause ids linked EN<->AR: {len(en_ids & ar_ids)}/{len(en_ids)}; glossary pairs from Article 1: {sum(1 for _, a in glossary if a)}")

        from falcon.policy.embed import Embedder

        emb = Embedder()
        if args.embed and not emb.enabled:
            raise SystemExit("--embed given but no embedding provider is configured (EMBEDDING_PROVIDER / OPENROUTER_API_KEY).")
        if args.no_embed or not emb.enabled:
            print("dense vectors: skipped" + ("" if args.no_embed else " (no embedding provider configured)")
                  + "; search_policy will report itself unavailable until the index is embedded")
        else:
            to_embed = [(row_id, r) for row_id, r in zip(ids, rows) if not r.title_only]
            vectors = emb.embed([embedding_text(r) for _, r in to_embed])
            cur.executemany("UPDATE policy.clauses SET embedding = %s::vector WHERE id = %s", [(str(v), row_id) for v, (row_id, _) in zip(vectors, to_embed)])
            conn.commit()
            print(f"embedded {len(to_embed)} rows with {emb.provider}/{emb.model} (dim {len(vectors[0])}); {len(rows) - len(to_embed)} title-only rows left without a vector")


if __name__ == "__main__":
    main()
