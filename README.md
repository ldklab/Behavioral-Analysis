# Behavioral Analysis of AI Code Generation Agents: Edit, Rewrite, and Repetition

> **Published at MSR '26 — 23rd International Conference on Mining Software Repositories**
> April 13–14, 2026 · Rio de Janeiro, Brazil · ACM
> DOI: [10.1145/3793302.3793605](https://doi.org/10.1145/3793302.3793605)

---

## Overview

This repository contains the full implementation for our MSR 2026 paper, which presents a large-scale behavioral study of five major AI code generation agents **Claude Code, Copilot, Cursor, Devin, and OpenAI Codex**  across real-world GitHub pull requests.

We investigate two core research questions:

- **RQ1: Edit or Rewrite?** Do agents tend to incrementally modify existing code, or do they substantially replace it?
- **RQ2: How repetitive is each agent?** Do agents generate diverse, novel code, or do they recycle patterns and token sequences?

Using token-level similarity metrics (Jaccard, TF-IDF cosine, fuzzy matching) and repetition analysis (n-gram distributions, Shannon entropy), we characterize the behavioral signatures of each agent across five programming languages: **TypeScript, Python, Go, Java, and C#**.

---

## Key Findings

### RQ1 — Edit vs. Rewrite Behavior

| Agent | Jaccard (Mean) | TF-IDF (Mean) | Fuzzy (Mean) | Tendency |
|---|---|---|---|---|
| Claude Code | 0.21 | 0.32 | 0.42 | **Rewriter** — lowest similarity, most aggressive replacement |
| Devin | 0.30 | 0.41 | 0.53 | **Editor** — highest similarity, incremental modification |
| Copilot | 0.26 | 0.37 | 0.45 | Intermediate |
| Cursor | 0.27 | 0.38 | 0.49 | Intermediate |
| OpenAI Codex | 0.27 | 0.36 | 0.47 | **Hybrid** — token-level vs. order-insensitive divergence |

> Claude Code systematically rewrites. Devin preserves. OpenAI Codex exhibits a unique pattern where order-insensitive token overlap substantially exceeds token-level similarity, suggesting a hybrid edit–rewrite strategy.

### RQ2 — Repetition and Token Diversity

| Agent | Unigram Rep. | 3-gram Rep. | 5-gram Rep. | Shannon Entropy |
|---|---|---|---|---|
| Claude Code | 0.83 | 0.45 | 0.31 | **6.62** (highest diversity) |
| Copilot | 0.82 | **0.48** | **0.36** | 6.09 |
| Cursor | 0.81 | 0.43 | 0.31 | 6.27 |
| Devin | 0.81 | 0.44 | 0.32 | 5.97 |
| OpenAI Codex | 0.76 | 0.36 | 0.25 | 5.63 (most consistent) |

> Claude Code repeats individual tokens but generates the most structurally diverse sequences (highest entropy). Copilot repeats full code patterns (highest n-gram repetition). OpenAI Codex is the most consistent and constrained generator.

---

## Dataset

We use the **[AIDev dataset](https://huggingface.co/datasets/hao-li/AIDev)** — a collection of real-world GitHub pull requests where code generation agents contribute to repositories with 100+ stars.

**After filtering:**

| Agent | PRs |
|---|---|
| OpenAI Codex | 16,088 |
| Devin | 3,335 |
| Copilot | 2,936 |
| Cursor | 898 |
| Claude Code | 277 |
| **Total** | **23,534** |

**Filtering criteria:**
- PRs with fewer than 50 added tokens excluded (prevent distortion of entropy/repetition measures)
- PRs above the 99th percentile of added-token count excluded (remove statistical outliers)
- Analysis restricted to top 5 languages: Go (30.1%), Python (21.4%), TypeScript (19.3%), C# (5.9%), Java (3.8%)

---

## Methodology

### RQ1 — Similarity Metrics (Edit vs. Rewrite)

We compute three complementary similarity metrics between **added** and **removed** code at the file level, then aggregate to PR level using a token-count weighted average (larger modifications contribute proportionally more):

```
Similarity(PR) = Σ (token_count_i / total_tokens) × similarity_i
```

**Jaccard Similarity** — set-based overlap of unique tokens:
```
Jaccard(A, R) = |tokens(A) ∩ tokens(R)| / |tokens(A) ∪ tokens(R)|
```

**TF-IDF Cosine Similarity** — vector-based, emphasizes discriminative tokens while downweighting common ones.

**Fuzzy Token-Set Similarity** — normalized approximate string matching, robust to token reordering and minor edits.

> Higher similarity → edit-like behavior. Lower similarity → rewrite-like behavior.

### RQ2 — Repetition Metrics

Applied only to **added code** within each PR (agent contributions, not surrounding human context):

- **Unigram repetition ratio** — frequency of individual token reuse
- **3-gram repetition ratio** — reuse of 3-token sequences
- **5-gram repetition ratio** — reuse of 5-token sequences (captures structural pattern memorization)
- **Shannon entropy** — overall vocabulary diversity; higher entropy = more diverse, less repetitive generation

---

## Repository Structure

```
Behavioral-Analysis/
│
├── data/
│   ├── README.md                   # Instructions for obtaining AIDev dataset
│   └── filtered_prs.csv            # Filtered PR metadata (after preprocessing)
│
├── src/
│   ├── preprocessing/
│   │   ├── filter_prs.py           # Apply token-count filtering
│   │   ├── tokenizer.py            # Language-agnostic tokenization
│   │   └── language_filter.py      # Restrict to top-5 languages
│   │
│   ├── rq1_similarity/
│   │   ├── jaccard.py              # Jaccard similarity computation
│   │   ├── tfidf_cosine.py         # TF-IDF cosine similarity computation
│   │   ├── fuzzy_similarity.py     # Fuzzy token-set similarity
│   │   └── aggregate.py            # Token-weighted PR-level aggregation
│   │
│   ├── rq2_repetition/
│   │   ├── ngram_repetition.py     # Unigram, 3-gram, 5-gram repetition ratios
│   │   └── shannon_entropy.py      # Shannon entropy computation
│   │
│   └── visualization/
│       ├── violin_plots.py         # Figure 1 — Edit vs. Rewrite violin plots
│       ├── boxplots_ngram.py       # Figure 2 — Agent repetitiveness comparison
│       └── entropy_violin.py       # Figure 3 — Shannon entropy distributions
│
├── notebooks/
│   ├── 01_preprocessing.ipynb      # Data loading, filtering, exploration
│   ├── 02_rq1_analysis.ipynb       # Full RQ1 similarity analysis
│   └── 03_rq2_analysis.ipynb       # Full RQ2 repetition analysis
│
├── results/
│   ├── table1_similarity.csv       # Table 1 — Similarity metrics per agent
│   ├── table2_repetition.csv       # Table 2 — Repetition metrics per agent
│   └── figures/                    # All paper figures (PDF + PNG)
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Git

### Installation

```bash
git clone https://github.com/ldklab/Behavioral-Analysis.git
cd Behavioral-Analysis
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
rapidfuzz>=3.0.0
nltk>=3.8.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
jupyter>=1.0.0
datasets>=2.0.0       # HuggingFace datasets (for AIDev)
tqdm>=4.64.0
```

### Dataset Setup

The AIDev dataset is available on HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("hao-li/AIDev")
```

Place the raw data in `data/raw/` before running preprocessing.

---

## Running the Analysis

### Step 1 — Preprocess and Filter

```bash
python src/preprocessing/filter_prs.py \
    --input data/raw/aidev_prs.csv \
    --output data/filtered_prs.csv \
    --min_tokens 50 \
    --outlier_percentile 99 \
    --languages go python typescript csharp java
```

### Step 2 — RQ1: Compute Similarity Metrics

```bash
python src/rq1_similarity/jaccard.py --input data/filtered_prs.csv
python src/rq1_similarity/tfidf_cosine.py --input data/filtered_prs.csv
python src/rq1_similarity/fuzzy_similarity.py --input data/filtered_prs.csv
python src/rq1_similarity/aggregate.py --output results/table1_similarity.csv
```

### Step 3 — RQ2: Compute Repetition Metrics

```bash
python src/rq2_repetition/ngram_repetition.py --input data/filtered_prs.csv
python src/rq2_repetition/shannon_entropy.py --input data/filtered_prs.csv \
    --output results/table2_repetition.csv
```

### Step 4 — Generate Figures

```bash
python src/visualization/violin_plots.py       # Figure 1
python src/visualization/boxplots_ngram.py     # Figure 2
python src/visualization/entropy_violin.py     # Figure 3
```

Or run the full pipeline interactively using the notebooks in `notebooks/`.

---

## Practical Implications

Based on our behavioral findings:

| Use Case | Recommended Agent |
|---|---|
| Large-scale refactoring / substantial rewrites | **Claude Code** (systematic rewriter, highest diversity) |
| Targeted bug fixes / incremental updates | **Devin** (strong edit preference, high token preservation) |
| Standardized, consistent code production | **OpenAI Codex** (most constrained and consistent generator) |
| Varied solution exploration | **Claude Code** (highest Shannon entropy) |

> ⚠️ Teams using **Copilot** should be aware of its elevated n-gram repetition tendency and may need to explicitly prompt for variety in pattern-sensitive tasks.

---

## Limitations

- Analysis restricted to repositories with 100+ GitHub stars — findings may not generalize to smaller or niche projects
- Dataset is imbalanced: OpenAI Codex comprises 68.4% of PRs; Claude Code only 1.2% — statistical confidence is lower for underrepresented agents
- Patch-level analysis does not account for contextual factors: code domain, project complexity, prompt formulations, or system configurations
- Token-level and n-gram metrics cannot distinguish *desirable* repetition (consistent API usage, style guides) from *undesirable* redundancy

---

## Citation

If you use this code or findings in your work, please cite:

```bibtex
@inproceedings{abazar2026behavioral,
  title     = {Behavioral Analysis of AI Code Generation Agents: Edit, Rewrite, and Repetition},
  author    = {Abazar, Mahdieh and Farahmand, Reyhaneh and Ginde, Gouri and Tan, Benjamin and De Carli, Lorenzo},
  note      = {Abazar and Farahmand contributed equally as first authors},
  booktitle = {Proceedings of the 23rd International Conference on Mining Software Repositories (MSR '26)},
  year      = {2026},
  month     = {April},
  location  = {Rio de Janeiro, Brazil},
  publisher = {ACM},
  doi       = {10.1145/3793302.3793605},
  isbn      = {979-8-4007-2474-9}
}
}
```

---

## Authors

<table>
  <tr>
    <td align="center"><b>Mahdieh Abazar*</b><br/>mahdieh.abazar@ucalgary.ca</td>
    <td align="center"><b>Reyhaneh Farahmand*</b><br/>reyhaneh.farahmand@ucalgary.ca</td>
    <td align="center">Gouri Ginde<br/>gouri.ginde@ucalgary.ca</td>
    <td align="center">Benjamin Tan<br/>benjamin.tan1@ucalgary.ca</td>
    <td align="center">Lorenzo De Carli<br/>lorenzo.decarli@ucalgary.ca</td>
  </tr>
</table>
* Both authors contributed equally to this research.

University of Calgary · Calgary, Alberta, Canada

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
