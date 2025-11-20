# Explainability from Vote Traces - Paper 12 Makefile
#
# Complete build system for vote trace analysis and explainability paper

# Configuration
PAPER_DIR = .
TRACES_JSONL = $(PAPER_DIR)/data/vote_traces.jsonl
FIGS_DIR = $(PAPER_DIR)/figs
TABLES_DIR = $(PAPER_DIR)/tables
VT_VOTES = $(TRACES_JSONL)
VT_TPLDIR = $(PAPER_DIR)/templates
VT_TABLE = $(TABLES_DIR)/vote_contrib_table.tex
VT_TABLE_SNR = $(TABLES_DIR)/vt_tables_snr.tex
VT_BINS ?= -10,-5,0,5,10,15
VT_SNR_KEY ?= snr_db
VT_PAD_EDGES ?= 0   # 1 to enable ±∞ end bins

# Python environment
PYTHON = python3

# Default dataset and classifier (can be overridden)
DATASET_FUNC ?= dummy
CLASSIFIER_SPEC ?= dummy

# Export for scripts
export DATASET_FUNC
export CLASSIFIER_SPEC

# Helper for pad edges flag
define PAD_FLAG
$(if $(filter 1 true yes on,$(VT_PAD_EDGES)),--pad-edges,)
endef

.PHONY: all traces figs tables-vt pdf press clean help xai-figs xai-prune

# Main targets
all: press

help:
	@echo "Vote Trace Explainability Paper - Enhanced Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  traces     - Generate vote trace data (JSONL)"
	@echo "  figs       - Create all figures from trace data"
	@echo "  tables-vt  - Render global + SNR-stratified contribution tables"
	@echo "  xai-figs   - Generate XAI plots (vote timelines, Shapley bars, heatmap)"
	@echo "  xai-prune  - Test attribution faithfulness via model pruning"
	@echo "  pdf        - Build LaTeX paper"
	@echo "  press      - Complete pipeline: traces → figs → tables → pdf"
	@echo "  clean      - Remove generated files"
	@echo "  test       - Run with dummy data"
	@echo ""
	@echo "Configuration:"
	@echo "  DATASET_FUNC=$(DATASET_FUNC)"
	@echo "  CLASSIFIER_SPEC=$(CLASSIFIER_SPEC)"
	@echo "  VT_BINS=$(VT_BINS)"
	@echo "  VT_SNR_KEY=$(VT_SNR_KEY)" 
	@echo "  VT_PAD_EDGES=$(VT_PAD_EDGES)"
	@echo ""
	@echo "Examples:"
	@echo "  make test                              # Test with dummy data"
	@echo "  make tables-vt VT_BINS=\"-20,-10,0,10,20\" VT_PAD_EDGES=1"
	@echo "  make press                             # Full pipeline"

# Test target with dummy data
test:
	@echo "🧪 Running test with dummy data..."
	$(MAKE) DATASET_FUNC=dummy CLASSIFIER_SPEC=dummy press
	@echo "✅ Test complete"

# Generate vote traces
traces: $(TRACES_JSONL)

$(TRACES_JSONL): scripts/run_vote_trace_eval.py code/explainability_utils.py
	@echo "🔍 Generating vote traces..."
	@echo "   Dataset: $(DATASET_FUNC)"
	@echo "   Classifier: $(CLASSIFIER_SPEC)"
	@mkdir -p $(PAPER_DIR)/data
	$(PYTHON) scripts/run_vote_trace_eval.py \
		--dataset $(DATASET_FUNC) \
		--classifier $(CLASSIFIER_SPEC) \
		--outdir $(PAPER_DIR)/data \
		--max 500 \
		--shapley 256
	@echo "✅ Vote traces ready"

# Generate figures
figs: $(FIGS_DIR)/.figures_done

$(FIGS_DIR)/.figures_done: $(TRACES_JSONL) scripts/gen_vote_trace_figs.py
	@echo "📊 Generating figures..."
	@mkdir -p $(FIGS_DIR)
	$(PYTHON) scripts/gen_vote_trace_figs.py \
		--data $(TRACES_JSONL) \
		--outdir $(FIGS_DIR) \
		--examples 6
	@touch $(FIGS_DIR)/.figures_done
	@echo "✅ Figures ready"

# === Explainability (Vote Traces) ===
tables-vt: $(VT_TABLE) $(VT_TABLE_SNR)

# XAI figures generation
xai-figs:
	@mkdir -p figs
	$(PYTHON) scripts/xai_figs.py --in $(VT_VOTES) --figdir figs --topk 10 --heatmap-n 50000
	@echo "✅ XAI figures generated in figs/"

# XAI pruning faithfulness evaluation  
xai-prune: $(VT_VOTES)
	$(PYTHON) scripts/xai_pruning_eval.py \
		--traces $(VT_VOTES) \
		--baseline_acc $${BASELINE_ACC:-0.85} \
		--out data/xai_pruning_summary.json
	@echo "✅ Pruning analysis completed"

$(VT_TABLE) $(VT_TABLE_SNR): $(TRACES_JSONL) $(VT_TPLDIR)/vote_contrib_table.tex.j2 $(VT_TPLDIR)/vt_top_contrib_snr.tex.j2 scripts/render_vote_tables.py
	@echo "📄 Rendering contribution tables..."
	@mkdir -p $(TABLES_DIR)
	$(PYTHON) scripts/render_vote_tables.py \
		--in $(VT_VOTES) \
		--out $(VT_TABLE) \
		--tpldir $(VT_TPLDIR) \
		--snr-key $(VT_SNR_KEY) \
		--bins='$(VT_BINS)' \
		--out-snr $(VT_TABLE_SNR) \
		--topk-snr 5 \
		$(call PAD_FLAG)
	@echo "✅ Tables ready: $(VT_TABLE) and $(VT_TABLE_SNR)"

# Build PDF
pdf: $(PAPER_DIR)/main_vote_traces.pdf

$(PAPER_DIR)/main_vote_traces.pdf: $(PAPER_DIR)/main_vote_traces.tex $(FIGS_DIR)/.figures_done $(VT_TABLE) $(VT_TABLE_SNR) $(PAPER_DIR)/refs.bib
	@echo "📖 Building LaTeX paper..."
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error main_vote_traces.tex > /dev/null || true
	cd $(PAPER_DIR) && bibtex main_vote_traces > /dev/null 2>&1 || true
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error main_vote_traces.tex > /dev/null || true
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error main_vote_traces.tex > /dev/null || true
	@if [ -f $(PAPER_DIR)/main_vote_traces.pdf ]; then \
		echo "✅ PDF generated: $(PAPER_DIR)/main_vote_traces.pdf"; \
		ls -lh $(PAPER_DIR)/main_vote_traces.pdf | awk '{print "📄 Size: " $$5}'; \
	else \
		echo "❌ PDF generation failed"; \
		exit 1; \
	fi

# Complete pipeline
press: traces figs tables-vt pdf
	@echo ""
	@echo "🎯 Paper 12: Explainability from Vote Traces - COMPLETE"
	@echo ""
	@echo "📁 Generated files:"
	@find $(PAPER_DIR) -name "*.pdf" -o -name "*.jsonl" -o -name "*.tex" | grep -E '\.(pdf|jsonl)$$|tables.*\.tex$$' | sort
	@echo ""
	@echo "🚀 Ready for submission!"

# Install dependencies
deps:
	@echo "📦 Installing Python dependencies..."
	pip install matplotlib jinja2 numpy
	@echo "✅ Dependencies installed"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf $(PAPER_DIR)/data/vote_traces.jsonl
	rm -rf $(FIGS_DIR)/*.pdf $(FIGS_DIR)/.figures_done
	rm -rf $(TABLES_DIR)/*.tex
	rm -rf $(PAPER_DIR)/*.pdf $(PAPER_DIR)/*.aux $(PAPER_DIR)/*.log $(PAPER_DIR)/*.bbl $(PAPER_DIR)/*.blg
	@echo "✅ Clean complete"

# Development utilities
debug-traces:
	@echo "🔍 Analyzing trace data..."
	@if [ -f $(TRACES_JSONL) ]; then \
		echo "Trace file size: $$(du -h $(TRACES_JSONL) | cut -f1)"; \
		echo "Number of traces: $$(wc -l < $(TRACES_JSONL))"; \
		echo "First trace sample:"; \
		head -1 $(TRACES_JSONL) | $(PYTHON) -m json.tool | head -20; \
	else \
		echo "No trace file found. Run 'make traces' first."; \
	fi

# Pre-commit hook integration
install-hooks:
	@echo "🔧 Installing pre-commit hooks..."
	@mkdir -p .git/hooks
	@echo '#!/usr/bin/env bash' > .git/hooks/pre-commit-vote-traces
	@echo 'set -euo pipefail' >> .git/hooks/pre-commit-vote-traces
	@echo 'changed=$$(git diff --cached --name-only | grep -E "^paper_Explainability_from_Vote_Traces/data/vote_traces\.jsonl$$" || true)' >> .git/hooks/pre-commit-vote-traces
	@echo 'if [ -n "$$changed" ]; then' >> .git/hooks/pre-commit-vote-traces
	@echo '  echo "🔎 vote_traces.jsonl changed; regenerating figures and tables..."' >> .git/hooks/pre-commit-vote-traces
	@echo '  make figs tables-vt || { echo "❌ regeneration failed"; exit 1; }' >> .git/hooks/pre-commit-vote-traces
	@echo '  git add paper_Explainability_from_Vote_Traces/figs paper_Explainability_from_Vote_Traces/tables' >> .git/hooks/pre-commit-vote-traces
	@echo 'fi' >> .git/hooks/pre-commit-vote-traces
	@chmod +x .git/hooks/pre-commit-vote-traces
	@echo "✅ Hooks installed"

# Check system readiness
check:
	@echo "🔍 Checking system readiness..."
	@$(PYTHON) --version
	@$(PYTHON) -c "import matplotlib; print('matplotlib: OK')" || echo "matplotlib: MISSING (run 'pip install matplotlib')"
	@$(PYTHON) -c "import jinja2; print('jinja2: OK')" || echo "jinja2: MISSING (run 'pip install jinja2')"
	@$(PYTHON) -c "import numpy; print('numpy: OK')" || echo "numpy: MISSING (run 'pip install numpy')"
	@which pdflatex > /dev/null && echo "pdflatex: OK" || echo "pdflatex: MISSING (install LaTeX distribution)"
	@echo "✅ System check complete"