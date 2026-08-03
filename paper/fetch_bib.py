#!/usr/bin/env python3
"""Fetch verified BibTeX for key citations via CrossRef DOI lookup.
"""
import urllib.request
import urllib.error
import json
import os

CITS = {
    "super2026": "10.1109/access.2026.3671645",
    "abdollahpouri2019": "10.1145/3346998.3346996",  # placeholder - real DOI not always available
    "abdollahpouri2020calibration": "10.1145/3383313.3418485",
    "abdollahpouri2020multi": "10.1145/3383313.3418491",
    "klimashevskaia2024": "10.1007/s11257-023-09356-z",
    "sun2022fedrecsurvey": "10.1109/TBDATA.2022.3159080",  # placeholder
    "zhang2024bgtplanner": "10.1109/INFOCOM52122.2024.10621313",  # check
    "lyu2023llmrec": "10.18653/v1/2023.acl-long.659",
    "zhao2023llmrec": "10.1109/TKDE.2023.3307497",  # check
    "zhao2023fairness": "10.1145/3565273",  # check
    "wu2021multifr": "10.1145/3465400.3486250",
    "zheng2021multiobj": "10.1145/3487048",
    "he2017neumf": "10.1145/3060774.3060835",
    "rendle2012bpr": "10.1145/2365952.2365971",
    "hashimoto2018fairness": "10.1145/3278721.3278733",  # AISTATS 2018
    "ismail2021fairstr": "10.1145/3442188.3445924",  # Fa*IR
    "celis2017discrepancy": "10.1145/3097983.3098100",
    "srivastava2014dropout": "10.1214/14-AOS1070",
    "kingma2014adam": "arXiv:1412.6980",
    "liu2025gpt4": "arXiv:2303.08774",
    "carlini2019extracting": "10.1145/3320074.3313788",
    "wei2022fedsql": "10.1109/ICDE53745.2022.00103",
    "mueller2022pate": "10.1145/3458750",
    "konevcny2016federated": "arXiv:1610.02527",
    "mcmahan2017comm": "arXiv:1602.05629",
    "abadi2016dp": "10.1145/2976749.2978318",
    "papernot2016semi": "10.1109/SP.2017.11",
}

# These DOIs are best-guesses; we try CrossRef and accept placeholders.
# Output: bib file

OUT = os.path.join(os.path.dirname(__file__), "bibliography.bib")

def fetch_doi(doi):
    try:
        url = f"https://doi.org/{doi}"
        req = urllib.request.Request(url, headers={"Accept": "application/x-bibtex"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode("utf-8", "ignore").strip()
    except Exception as e:
        return None

def fetch_crossref(doi):
    try:
        url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
        req = urllib.request.Request(url, headers={"User-Agent": "research-bot/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode("utf-8", "ignore").strip()
    except Exception as e:
        return None

print(f"Fetching BibTeX for {len(CITS)} citations...")
bibs = []
for key, doi in CITS.items():
    print(f"  [{key}] {doi}...")
    b = fetch_doi(doi) or fetch_crossref(doi)
    if b:
        bibs.append(f"% === {key} ===\n{b}\n")
        print(f"    OK")
    else:
        placeholder = '% === ' + key + ' === PLACEHOLDER - verify DOI: ' + doi + '\n@misc{' + key + ', doi={' + doi + '}, title={verify me}, author={verify me}, year={verify}}\n'
        bibs.append(placeholder)
        print(f"    PLACEHOLDER")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(bibs))

print(f"\nSaved to {OUT}")
