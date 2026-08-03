"""Try multiple sources to locate the SUPER paper metadata + abstract."""
import json, urllib.parse, urllib.request, sys

TITLE = "SUPER Smart User-centric Popularity Exposure Reduction for Fair and Diverse Recommendations"

# 1. Semantic Scholar direct API (graph endpoint, no key needed for low-rate)
def s2_search(q, n=5):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode({
        "query": q, "limit": n,
        "fields": "title,authors,year,abstract,venue,externalIds,url,citationCount",
    })
    print("S2:", url)
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print("S2 error:", e)
        return {"data": []}

# 2. Crossref by title
def crossref(q):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({
        "query.title": q, "rows": 5
    })
    print("CR:", url)
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print("CR error:", e)
        return {"message": {"items": []}}

s2 = s2_search(TITLE)
items = s2.get("data", [])
print("\n=== S2 hits:", len(items))
for it in items:
    print("\nTITLE:", it.get("title"))
    print("YEAR:", it.get("year"), "VENUE:", it.get("venue"))
    print("URL:", it.get("url"))
    print("AUTHORS:", [a.get("name") for a in (it.get("authors") or [])])
    print("CITES:", it.get("citationCount"))
    ab = (it.get("abstract") or "")
    print("ABSTRACT:", ab[:800])
    print("IDS:", it.get("externalIds"))

cr = crossref(TITLE)
items_cr = cr.get("message", {}).get("items", [])
print("\n\n=== Crossref hits:", len(items_cr))
for it in items_cr:
    print("\nTITLE:", it.get("title", [""])[0])
    print("YEAR:", (it.get("published-print") or {}).get("date-parts", [["?"]])[0][0])
    print("DOI:", it.get("DOI"))
    print("CONT:", (it.get("container-title") or [""])[0] if it.get("container-title") else "")
    print("URL:", it.get("URL") or it.get("link"))
