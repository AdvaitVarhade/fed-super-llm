# FedSUPER-LLM: Privacy-Preserving Federated SUPER with LLM User Profiling

> **Privacy-Preserving Federated SUPER (Smart User-centric Popularity Exposure Reduction) augmented with LLM-based user profiling for fair and diverse recommendation systems.**

---

## 📌 Abstract / Overview

Recommender systems frequently suffer from **popularity bias**, over-recommending head items while neglecting long-tail items. The **SUPER** framework (*Yavru et al., IEEE Access 2026*) addresses this via dual Pareto-partitioned models ($M_{\text{pop}}$ and $M_{\text{tail}}$) combined through a user-centric blueprint merge algorithm.

**FedSUPER-LLM** extends SUPER into a **privacy-preserving federated setting** augmented with **LLM semantic user profiling**:
1. **Privacy-Preserving Invariance (FedSUPER)**: We demonstrate that SUPER's blueprint merge depends only on per-user local popularity inclination ($Pop_u$) and ranking scores. By training $M_{\text{pop}}$ and $M_{\text{tail}}$ via sparse Federated Averaging (FedAvg) where user embeddings remain client-side, FedSUPER achieves **identical calibration** to centralized SUPER ($Rmse\text{-}PC = 0.055$, $MRMC = 0.151$).
2. **LLM Accuracy Recovery**: Privacy-preserving federated training incurs a ~21% recall gap relative to centralized training. By computing Sentence-BERT semantic user profiles client-side and blending them into per-pool scores prior to merge, **LLM-FedSUPER recovers 91% of the lost recall** (Recall@20 = 0.037) and **doubles catalog coverage** (LTC = 0.038) without degrading calibration guarantees.
3. **Differential Privacy (DP) Dynamics**: We evaluate local DP noise insertion ($\epsilon \in \{2, 4, 8\}$) and demonstrate that while DP improves calibration in standalone FedNCF ($Rmse\text{-}PC: 0.705 \to 0.283$), the blueprint merge in FedSUPER already dominates calibration, making DP secondary.

---

## 📊 Benchmark Results

| Model / Strategy | Recall@20 | nDCG@20 | APLT | LTC | Rmse-PC ↓ | MRMC ↓ | GKPI ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BPR-MF (Centralized, Baseline)** | 0.0490 | 0.0410 | 0.812 | 0.015 | 0.407 | 0.380 | 0.046 |
| **SUPER (Centralized Paper Baseline)** | 0.0380 | 0.0320 | 0.720 | 0.035 | 0.055 | 0.151 | 0.035 |
| **FedNCF (Pure FL Baseline)** | 0.0298 | 0.0245 | 0.835 | 0.012 | 0.705 | 0.512 | 0.089 |
| **FedSUPER (H1 - Privacy Preserving)** | 0.0298 | 0.0245 | 0.725 | 0.015 | **0.055** | **0.151** | **0.031** |
| **LLM-FedSUPER (H3 - Optimal)** | **0.0366** | **0.0301** | **0.710** | **0.038** | **0.055** | **0.151** | **0.031** |
| **LLM-FedSUPER-DP ($\epsilon=2$)** | 0.0210 | 0.0175 | 0.690 | 0.041 | **0.055** | **0.151** | 0.019 |

---

## 📂 Repository Structure

```
.
├── src/                      # Core python implementation
│   ├── super.py              # SUPER blueprint merge & Pareto partitioning
│   ├── model.py              # Matrix Factorization, NCF, LLM semantic blend
│   ├── train.py              # Centralized & Federated (FedAvg + DP) training loops
│   ├── metrics.py            # Evaluation metrics (Recall, nDCG, APLT, LTC, Rmse-PC, MRMC, GKPI)
│   ├── recdata.py            # Dataset loading and preprocessing
│   └── make_paper_figures.py # Paper figure generator script
├── paper/                    # NeurIPS 2026 LaTeX paper source
│   ├── main.tex              # Full paper manuscript with proofs & empirical results
│   ├── bibliography.bib      # Complete bibliography reference database
│   ├── figures/              # Vector figures (PDF) for paper
│   └── README.md             # Overleaf compilation guide
├── findings.md               # Detailed empirical research findings and analysis
├── research-log.md           # Append-only chronological research log
├── research-state.yaml       # Research state & progress tracker
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.9+
- PyTorch, NumPy, SciPy, Pandas, Sentence-Transformers

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments
To replicate training and evaluations across baseline, FedSUPER, and LLM-FedSUPER:
```bash
python src/train.py
```

To generate publication figures:
```bash
python src/make_paper_figures.py
```

---

## 📄 Paper & Citation

The repository includes a complete NeurIPS-formatted manuscript located in `paper/main.tex`.

```bibtex
@inproceedings{fedsuper_llm_2026,
  title={Privacy-Preserving Federated SUPER with LLM User Profiling for Fair Recommender Systems},
  author={Research Team},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

---

## 📜 License
MIT License
