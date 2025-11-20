# Explainability from Vote Traces in RF Ensembles

A comprehensive system for adding interpretability to RF signal classification ensembles through vote tracing, Shapley attribution, and visual analysis.

## Overview

This system provides:
- **Audit hooks** for recording ensemble decision processes
- **Shapley-like attribution** for quantifying model contributions  
- **Visual analysis tools** for vote timelines and contribution patterns
- **Automated table generation** for publication-ready results
- **Complete LaTeX paper** with IEEE formatting

## Quick Start

```bash
# Test with dummy data
make test

# Use with your own data
DATASET_FUNC="my_dataset:iter_eval" CLASSIFIER_SPEC="my_classifier:MyEnsemble" make press

# Individual components
make traces      # Generate vote trace data
make figs        # Create visualizations  
make tables-vt   # Render contribution tables
make pdf         # Build LaTeX paper
```

## System Architecture

```
Raw Dataset → Vote Traces → Analysis → Paper
     ↓             ↓          ↓         ↓
  Dataset     JSONL Logs   Figures   LaTeX PDF
  Iterator    + Shapley    + Tables
```

### Core Components

1. **Audit Hooks** (`code/audit_hooks_patch.py`)
   - Lightweight instrumentation for `classify_signal()`
   - Records per-model logits, probabilities, weights, temperatures
   - Captures timing, OSR decisions, and aggregate statistics

2. **Explainability Utilities** (`code/explainability_utils.py`)
   - Shapley-like attribution using Monte Carlo sampling
   - Model agreement analysis and disagreement detection
   - Text explanation generation

3. **Data Pipeline** (`scripts/run_vote_trace_eval.py`)
   - Processes datasets with configurable classifiers
   - Generates structured JSONL trace logs
   - Computes Shapley contributions for each signal

4. **Visualization** (`scripts/gen_vote_trace_figs.py`)
   - Vote timeline plots (per-model probabilities)
   - Shapley contribution bar charts
   - Model agreement matrices and distribution analysis

5. **Table Generation** (`scripts/render_vote_tables.py`)
   - Top contributing models with statistics
   - Jinja2 templating for LaTeX tables
   - Automated ranking and formatting

## File Structure

```
paper_Explainability_from_Vote_Traces/
├── code/                           # Core algorithms
│   ├── explainability_utils.py    # Shapley attribution
│   └── audit_hooks_patch.py       # Classification instrumentation
├── scripts/                       # Data processing
│   ├── run_vote_trace_eval.py     # Dataset evaluation runner
│   ├── gen_vote_trace_figs.py     # Figure generation
│   └── render_vote_tables.py      # Table rendering
├── templates/                     # LaTeX templates
│   └── vote_contrib_table.tex.j2  # Contribution table template
├── data/                          # Generated data
│   └── vote_traces.jsonl         # Vote trace logs
├── figs/                          # Generated figures
│   ├── vote_timeline_*.pdf        # Per-signal timelines
│   ├── vote_shapley_*.pdf         # Contribution bars
│   └── vote_shapley_mean.pdf      # Aggregate analysis
├── tables/                        # Generated tables
│   └── vote_contrib_table.tex     # LaTeX contribution table
├── main_vote_traces.tex           # Main paper document
├── refs.bib                       # Bibliography
└── Makefile                       # Build system
```

## Integration with Existing Code

### Option 1: Decorator Approach
```python
from audit_hooks_patch import add_vote_trace_hooks

@add_vote_trace_hooks
def classify_signal(self, signal, override_temperature=None):
    # Your existing classification logic
    return prediction, confidence, probabilities
```

### Option 2: Manual Integration
See `manual_audit_hooks_example()` in `code/audit_hooks_patch.py` for the complete code to insert into your `classify_signal()` method.

### Option 3: Automatic Patching
```python
from audit_hooks_patch import patch_ensemble_classifier_file
patch_ensemble_classifier_file("path/to/ensemble_ml_classifier.py")
```

## Generated Outputs

### Vote Traces (JSONL)
Each line contains:
```json
{
  "id": "signal_001",
  "true": 2,
  "pred": 1, 
  "pmax": 0.78,
  "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
  "temperatures": [1.0, 1.2, 0.8, 1.0, 1.1],
  "per_model_pmax": [0.65, 0.78, 0.71, 0.69, 0.58],
  "shapley_top1": [0.12, 0.31, -0.05, 0.18, 0.09],
  "trace": { /* full event details */ }
}
```

### Figures
- **Vote Timelines**: Per-model confidence vs ensemble decision
- **Shapley Bars**: Attribution for individual signals  
- **Mean Contributions**: Dataset-wide model importance
- **Agreement Matrix**: Pairwise model consensus analysis
- **Distribution Plots**: Statistical summary of contributions

### Tables
- **Top Contributing Models**: Ranked by mean Δp with statistics
- **Model Performance Summary**: Confidence, entropy, timing metrics

## Configuration

### Environment Variables
```bash
export DATASET_FUNC="my_dataset_module:iter_eval"     # Dataset iterator function
export CLASSIFIER_SPEC="ensemble_ml_classifier:EnsembleMLClassifier"  # Classifier class
```

### Makefile Targets
```bash
make help        # Show available commands
make check       # Verify system dependencies
make deps        # Install Python dependencies
make clean       # Remove generated files
make debug-traces # Analyze trace data
```

## Dependencies

### Required
- Python 3.8+
- matplotlib (figures)
- jinja2 (tables)  
- numpy (computations)
- LaTeX distribution (PDF generation)

### Installation
```bash
make deps        # Install Python packages
# LaTeX: sudo apt-get install texlive-full (Ubuntu)
```

## Applications

### Model Debugging
- Identify consistently underperforming ensemble members
- Find signals where models systematically disagree
- Detect unusual voting patterns (potential adversarial inputs)

### Dataset Analysis  
- Quantify per-class ensemble behavior
- Find challenging signals requiring manual review
- Validate ensemble composition and weighting

### Confidence Calibration
- Analyze relationship between ensemble confidence and accuracy
- Identify over-confident vs under-confident regions
- Guide threshold selection for open-set recognition

### Research Applications
- Generate explainability datasets for further analysis
- Support ensemble architecture design decisions  
- Enable reproducible interpretability studies

## Paper Generation

The system generates a complete IEEE-format paper including:
- Technical methodology sections
- Comprehensive figure gallery
- Quantitative analysis tables
- Reproducibility instructions
- Complete bibliography

### Customization
- Modify `main_vote_traces.tex` for content changes
- Update `templates/*.j2` for table formatting
- Adjust `scripts/*.py` for analysis parameters

## Performance

### Overhead
- Audit hooks: ~0.1-0.5ms per signal
- Shapley computation: O(M × S) where M = models, S = samples  
- Memory: ~1-2KB per signal for trace storage
- Figure generation: ~10-50ms per plot

### Scalability
- Supports 1K-100K+ signals efficiently
- Configurable Shapley sampling (default: 256)
- Streaming-compatible for real-time analysis
- Parallel processing for batch workloads

## Citation

```bibtex
@article{gilbert2025explainability,
  title={Explainability from Vote Traces in RF Ensembles},
  author={Gilbert, Benjamin J.},
  journal={IEEE Transactions on Signal Processing},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues:
1. Check `make help` for available commands
2. Run `make check` to verify system setup
3. Review example outputs in the `test` target
4. Examine trace data with `make debug-traces`
