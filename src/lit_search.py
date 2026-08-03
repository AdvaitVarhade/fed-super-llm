"""Bootstrap literature search using Semantic Scholar + arXiv.

Saves summaries into literature/ folder, plus a running survey.md.
"""
import os
import json
import time
import datetime as dt
import re

from semanticscholar import SemanticScholar

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIT = os.path.join(ROOT, "literature")
os.makedirs(LIT, exist_ok=True)

QUERIES = [
    ("popularity_bias",
     "popularity bias recommender systems fairness exposure"),
    ("super_paper",
     "SUPER Smart User-centric Popularity Exposure Reduction recommendations"),
    ("federated_rec",
     "federated learning recommender systems privacy"),
    ("dp_rec",
     "differential privacy recommender federated"),
    ("llm_rec",
     "LLM large language model user profiling recommendation"),
    ("multi_obj_fairness",
     "multi-objective optimization fairness diversity novelty recommender"),
]

def safe_slug(s, maxlen=80):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:maxlen]

sch = SemanticScholar()

def search_sem(query, n=8):
    try:
        res = sch.search_paper(query, limit=n)
        out = []
        for it in res:
            try:
                out.append({
                    "title": it.title,
                    "authors": ", ".join(a.name for a in (it.authors or [])[:4]),
                    "year": it.year,
                    "abstract": (it.abstract or "")[:1500],
                    "url": it.url,
                    "paperId": getattr(it, "paperId", None),
                    "venue": getattr(it, "venue", "") or "",
                    "citationCount": getattr(it, "citationCount", None),
                })
            except Exception as e:
                print("iter error", e)
        return out
    except Exception as e:
        print(f"search_sem error for '{query}': {e}")
        return []

all_entries = {}
for tag, q in QUERIES:
    print(f"\n=== {tag}: {q}")
    hits = search_sem(q, n=8 if tag != "super_paper" else 5)
    print(f"  got {len(hits)} hits")
    all_entries[tag] = hits
    time.sleep(2)

# Save individual paper summaries + survey.md entries
survey_lines = ["# Literature Survey — SUPER Extension\n",
                "Project: Privacy-Preserving Fair/Diverse Recommendations\n",
                f"Generated: {dt.datetime.now().isoformat()}\n",
                "\n"]

for tag, hits in all_entries.items():
    survey_lines.append(f"\n## Cluster: {tag}  ({len(hits)} papers)\n")
    for h in hits:
        if not h["title"]:
            continue
        slug = safe_slug(h["title"])
        path = os.path.join(LIT, f"{tag}__{slug}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(h, f, indent=2, ensure_ascii=False)
        survey_lines.append(f"- **{h['title']}** ({h['authors']}, {h['year']}, {h['venue']})  ")
        survey_lines.append(f"  {h['abstract'][:300]}")
        survey_lines.append(f"  URL: {h['url']}\n")

with open(os.path.join(LIT, "survey.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(survey_lines))

# Save consolidated raw json
with open(os.path.join(LIT, "_raw_hits.json"), "w", encoding="utf-8") as f:
    json.dump(all_entries, f, indent=2, ensure_ascii=False)

# Find & flag the SUPER paper specifically
super_hits = all_entries.get("super_paper", [])
main = None
for h in super_hits:
    t = (h["title"] or "").lower()
    if "super" in t and ("popularity" in t or "exposure" in t):
        main = h
if main is None and super_hits:
    main = super_hits[0]
print("\nSUPER candidate:", main)
print(f"Total summarized: {sum(len(v) for v in all_entries.values())}")
print("Saved to:", LIT)
