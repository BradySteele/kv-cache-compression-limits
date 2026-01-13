# On the Limits of Learned Importance Scoring for KV Cache Compression

**A Negative Results Paper**

This repository contains the code and experimental artifacts for our study demonstrating that learned importance scoring fails to outperform simple heuristics for KV cache compression in large language models.

---

## Key Findings

| Retention | Best Method | SIP Rank | Statistical Significance |
|-----------|-------------|----------|--------------------------|
| 10% | Position-Heuristic | 4th of 6 | Heuristic significantly better (p < 0.01) |
| 25% | Position-Heuristic | 4th of 6 | Heuristic significantly better (p < 0.01) |
| 50% | Prefill-Attn | 5th of 6 | Prefill significantly better (p < 0.05) |
| 75% | Prefill-Attn | 4th of 6 | Prefill significantly better (p < 0.05) |

**SIP vs. Random**: No statistically significant difference at any retention level (95% CI overlaps for all comparisons).

### Practical Recommendations

Based on our evaluation:

- **Aggressive compression (10-25% retention)**: Use position-based heuristics (keep first 4 + last N tokens)
- **Moderate compression (50-75% retention)**: Use prefill attention scores
- **Light compression (>75% retention)**: Any method suffices; all converge to similar performance

**We recommend against using learned importance scoring** -- it adds model complexity without performance benefit.

---

## Evaluation Methodology

All results are based on rigorous statistical evaluation:

- **5 random seeds** per configuration
- **95% confidence intervals** reported for all metrics
- **Paired t-tests** for method comparisons
- **Multiple retention levels** (10%, 25%, 50%, 75%)
- **Controlled experimental conditions** with fixed hyperparameters across methods

---

## Repository Structure

```
kv-cache-compression-limits/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── notebooks/
│   └── evaluation_experiments.ipynb  # Full experimental evaluation
└── results/
    └── figures/                   # Generated paper figures (PDF)
        ├── training_convergence.pdf
        ├── calibration_curve.pdf
        ├── attention_paradox.pdf
        ├── forest_plot_ci.pdf
        ├── information_analysis.pdf
        ├── significance_matrix.pdf
        └── sip_vs_random.pdf
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/kv-cache-compression-limits.git
cd kv-cache-compression-limits

# Create virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- See `requirements.txt` for complete dependency list

---

## Usage

### Running the Full Evaluation

The complete experimental evaluation is contained in the Jupyter notebook:

```bash
jupyter notebook notebooks/evaluation_experiments.ipynb
```

This notebook includes:
- Model loading and configuration
- Implementation of all baseline methods (Random, Position-Heuristic, Prefill-Attn)
- Implementation of published methods (H2O, Scissorhands, StreamingLLM, SnapKV)
- SIP (Speculative Importance Predictor) training and evaluation
- Statistical analysis with confidence intervals and significance testing
- Results aggregation across seeds and retention levels

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
