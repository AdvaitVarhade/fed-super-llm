"""Bootstrap literature search using arXiv (no key needed).

Saves per-paper notes and a running survey.md. Saves after each cluster so
partial progress is preserved if Semantic Scholar API is slow.
"""
import os, json, time, datetime as dt, re, urllib.parse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIT = os.path.join(ROOT, "literature")
os.makedirs(LIT, exist_ok=True)

import urllib.request, xml.etree.ElementTree as ET

QUERIES = [
    ("popularity_bias",
     "(ti:popularity OR ti:exposure OR ti:bias) AND (ti:recommender OR abs:recommender)"),
    ("super_popularity_exposure",
     "(all:SUPER AND all:popularity AND all:recommend) OR (all:'popularity exposure' AND all:recommend)"),
    ("federated_rec",
     "(ti:federated AND (ti:recommend OR abs:recommend))"),
    ("dp_rec",
     "(all:differential-privacy AND all:recommender)"),
    ("llm_rec",
     "(ti:large-language-model OR ti:LLM OR abs:LLM) AND (ti:recommend OR abs:recommend)"),
    ("multi_obj_fair",
     "((ti:multi-objective OR ti:fairness OR ti:diversity) AND (ti:recommend OR abs:recommend))"),
]

NS = {"a": "http://www.w3.org/2005/Atom"}

def safe_slug(s, maxlen=80):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:maxlen]

def search_arxiv(query, n=12, timeout=40):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query, "start": 0, "max_results": n,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    out = []
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read().decode("utf-8", "ignore")
        root = ET.fromstring(data)
        for e in root.findall("a:entry", NS):
            t = e.find("a:title", NS).text.strip().replace("\n", " ")
            t = re.sub(r"\s+", " ", t)
            abs_ = (e.find("a:summary", NS).text or "").strip()
            abs_ = re.sub(r"\s+", " ", abs_)[:1500]
            authors = ", ".join(a.find("a:name", NS).text for a in e.findall("a:author", NS))
            link = e.find("a:id", NS).text
            pub = (e.find("a:published", NS).text or "")[:10]
            out.append({"title": t, "authors": authors, "year": pub[:4], "abstract": abs_, "url": link, "venue": "arXiv"})
    except Exception as e:
        print(f"arxiv error for '{query}': {e}")
    return out

all_entries = {}
raw_path = os.path.join(LIT, "_raw_arxiv.json")
existing = {}
if os.path.exists(raw_path):
    with open(raw_path, encoding="utf-8") as f:
        existing = json.load(f)
survey_path = os.path.join(LIT, "survey.md")
survey_lines = []
start_fresh = not os.path.exists(survey_path)
if start_fresh:
    survey_lines = ["# Literature Survey — SUPER Extension (arXiv)\n",
                    f"Generated: {dt.datetime.now().isoformat()}\n"]

for tag, q in QUERIES:
    if tag in existing and existing[tag]:
        print(f"\n=== {tag}: already have {len(existing[tag])}, skipping")
        all_entries[tag] = existing[tag]
        continue
    print(f"\n=== {tag}: {q}", flush=True)
    hits = search_arxiv(q, n=12)
    print(f"  got {len(hits)} hits", flush=True)
    all_entries[tag] = hits
    survey_lines.append(f"\n## Cluster: {tag}  ({len(hits)} papers)\n")
    for h in hits:
        slug = safe_slug(h["title"])
        path = os.path.join(LIT, f"{tag}__{slug}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(h, f, indent=2, ensure_ascii=False)
        survey_lines.append(f"- **{h['title']}** ({h['authors'][:120]}, {h['year']}, {h['venue']})  ")
        survey_lines.append(f"  {h['abstract'][:300]}")
        survey_lines.append(f"  URL: {h['url']}\n")

    # save after each cluster so partial progress is preserved
    with open(survey_path, "w", encoding="utf-8") as f:
        f.write("\n".join(survey_lines))
    with open(os.path.join(LIT, "_raw_arxiv.json"), "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    time.sleep(3)

print(f"\nTotal summarized: {sum(len(v) for v in all_entries.values())}")
print("Saved to:", LIT)
