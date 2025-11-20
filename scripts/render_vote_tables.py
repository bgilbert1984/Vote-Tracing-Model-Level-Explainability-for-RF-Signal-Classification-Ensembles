#!/usr/bin/env python3
"""
Vote Contribution Table Renderer with SNR Stratification

Generates LaTeX tables of top contributing models with Shapley attribution statistics.
Supports both global and SNR-stratified analysis with configurable bins and ±∞ edge padding.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from collections import defaultdict
from typing import List, Dict, Any, Tuple

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Warning: jinja2 not available, install with: pip install jinja2")


def load_signals(json_paths: List[str]):
    """Load signals from JSON files with flexible format support."""
    for p in json_paths:
        data = json.loads(Path(p).read_text())
        # Accept either a list of per-signal records, or a dict with key "signals"
        items = data["signals"] if isinstance(data, dict) and "signals" in data else data
        for s in items:
            yield s


def aggregate_top_contrib(signals) -> Tuple[int, List[Dict[str, Any]]]:
    """Compute global top contributing models statistics."""
    per_model_vals = defaultdict(list)
    top_count = defaultdict(int)
    N = 0
    
    for s in signals:
        contrib = s.get("shapley_contribution") or s.get("metadata", {}).get("shapley_contribution")
        if not contrib:
            continue
        N += 1
        
        # Collect contribution values for each model
        for m, v in contrib.items():
            try:
                per_model_vals[m].append(float(v))
            except Exception:
                pass
        
        # Count which model had the maximum contribution
        m_top = max(contrib.items(), key=lambda kv: kv[1])[0]
        top_count[m_top] += 1
    
    # Compute statistics
    rows = []
    for m, vals in per_model_vals.items():
        if not vals:
            continue
        mean_dp = mean(vals)
        std_dp = pstdev(vals) if len(vals) > 1 else 0.0
        top_share = (top_count[m] / N) if N else 0.0
        rows.append({
            "model": m,
            "mean_dp": mean_dp,
            "std_dp": std_dp,
            "top_share": top_share
        })
    
    # Sort by mean contribution descending
    rows.sort(key=lambda r: r["mean_dp"], reverse=True)
    return N, rows


def parse_bins(bins_str: str, pad_edges: bool) -> List[Tuple[float, float]]:
    """
    Parse comma-separated bin edges into closed-open intervals.
    
    Args:
        bins_str: comma-separated edges, e.g. "-10,-5,0,5,10,15"
        pad_edges: if True, add (-inf, first) and (last, +inf) bins
        
    Returns:
        List of (lo, hi) bin intervals
    """
    edges = [float(x.strip()) for x in bins_str.split(",") if x.strip() != ""]
    if len(edges) < 1:
        raise ValueError("Need at least one edge. Provide 'a,b' for at least one bin.")
    
    # Create inner bins [a,b), [b,c), ...
    bins = list(zip(edges[:-1], edges[1:])) if len(edges) > 1 else []
    
    if pad_edges:
        # Add underflow (-inf, first) and overflow (last, +inf) bins
        bins = [(-math.inf, edges[0])] + bins + [(edges[-1], math.inf)]
    
    return bins


def _fmt_edge(x: float) -> str:
    """Format edge value for LaTeX display."""
    if math.isinf(x):
        return r"\infty"
    # Display integer values without decimal point
    return f"{int(x)}" if float(x).is_integer() else f"{x:g}"


def bin_label(lo: float, hi: float) -> str:
    """
    Generate pretty LaTeX label for closed-open bins.
    Infinite bounds render as $(-\\infty, a)$ and $[b, \\infty)$.
    """
    if math.isinf(lo) and lo < 0:
        return f"$(-\\infty,{_fmt_edge(hi)})$"
    if math.isinf(hi) and hi > 0:
        return f"$[{_fmt_edge(lo)},\\infty)$"
    return f"$[{_fmt_edge(lo)},{_fmt_edge(hi)})$"


def aggregate_snr_top_contrib(signals, snr_key: str, bins: List[Tuple[float, float]], topk: int = 5) -> List[Dict[str, Any]]:
    """
    Compute SNR-stratified top contributing models for each bin.
    
    Args:
        signals: Signal iterator
        snr_key: Key name for SNR values in signal records
        bins: List of (lo, hi) bin intervals
        topk: Maximum number of top models to include per bin
        
    Returns:
        List of bin statistics with top contributing models
    """
    bin_stats = []
    
    for (lo, hi) in bins:
        per_model_vals = defaultdict(list)
        top_count = defaultdict(int)
        N = 0
        
        for s in signals:
            # Extract SNR value from signal record or metadata
            meta = s.get("metadata", {})
            snr_val = s.get(snr_key, meta.get(snr_key, None))
            if snr_val is None:
                continue
                
            try:
                snr = float(snr_val)
            except Exception:
                continue
                
            # Check if SNR falls within current bin (closed-open intervals)
            if math.isinf(lo):
                in_bin = snr < hi
            elif math.isinf(hi):
                in_bin = snr >= lo
            else:
                in_bin = lo <= snr < hi
                
            if not in_bin:
                continue
            
            # Extract Shapley contributions
            contrib = s.get("shapley_contribution") or meta.get("shapley_contribution")
            if not contrib:
                continue
                
            N += 1
            
            # Collect contribution values
            for m, v in contrib.items():
                try:
                    per_model_vals[m].append(float(v))
                except Exception:
                    pass
            
            # Track top contributor for this signal
            m_top = max(contrib.items(), key=lambda kv: kv[1])[0]
            top_count[m_top] += 1
        
        # Compute statistics for this bin
        rows = []
        for m, vals in per_model_vals.items():
            if not vals:
                continue
            mean_dp = mean(vals)
            std_dp = pstdev(vals) if len(vals) > 1 else 0.0
            top_share = (top_count[m] / N) if N else 0.0
            rows.append({
                "model": m,
                "mean_dp": mean_dp,
                "std_dp": std_dp,
                "top_share": top_share
            })
        
        # Sort by mean contribution and limit to topk
        rows.sort(key=lambda r: r["mean_dp"], reverse=True)
        if topk and topk > 0:
            rows = rows[:topk]
        
        bin_stats.append({
            "label": bin_label(lo, hi),
            "N": N,
            "rows": rows
        })
    
    return bin_stats


def render_jinja(tpl_dir: Path, tpl_name: str, context: Dict[str, Any], out_tex: Path):
    """Render Jinja2 template to LaTeX file."""
    if not JINJA2_AVAILABLE:
        # Fallback manual generation for critical tables
        generate_table_manual(context, out_tex, tpl_name)
        return
    
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template(tpl_name)
    tex = tpl.render(**context)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex)
    print(f"✅ Wrote {out_tex}")


def generate_table_manual(context: Dict[str, Any], out_path: Path, template_type: str):
    """Manual LaTeX generation fallback when Jinja2 unavailable."""
    if template_type == "vt_top_contrib.tex.j2":
        # Global table
        rows = context["rows"]
        N = context["N"]
        latex = f"""% Auto-generated table. Do not edit by hand.
\\begin{{table}}[t]
\\centering
\\caption{{Top contributing models (mean $\\Delta p$) over {N} samples.}}
\\label{{tab:vote_contrib_top}}
\\begin{{tabular}}{{l c c c}}
\\toprule
Model & Mean $\\Delta p$ & Std & Top-Share \\\\
\\midrule
"""
        for r in rows:
            latex += f"{r['model']} & {r['mean_dp']:.4f} & $\\pm$ {r['std_dp']:.4f} & {100.0*r['top_share']:.1f}\\% \\\\\n"
        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    else:
        # SNR table fallback
        latex = "% Manual fallback not implemented for SNR table\n"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)


def main():
    """Main table rendering function."""
    ap = argparse.ArgumentParser(description="Render vote contribution tables")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="JSON files with per-signal shapley_contribution dicts")
    ap.add_argument("--out", default="tables/vt_tables.tex",
                    help="Output path for global table")
    ap.add_argument("--tpldir", default="templates",
                    help="Directory containing Jinja2 templates")
    
    # SNR-stratified table options
    ap.add_argument("--snr-key", default=None,
                    help="Key for SNR in each record or record.metadata")
    ap.add_argument("--bins", default=None,
                    help='Comma-separated bin edges, e.g. "-10,-5,0,5,10,15"')
    ap.add_argument("--out-snr", default="tables/vt_tables_snr.tex",
                    help="Output path for SNR-stratified table")
    ap.add_argument("--topk-snr", type=int, default=5,
                    help="Top-K models per bin (default 5)")
    ap.add_argument("--pad-edges", action="store_true",
                    help="Wrap ends with (-inf, first) and [last, +inf) bins and label with ±∞")
    
    args = ap.parse_args()
    
    # Check input files exist
    missing_files = [f for f in args.inputs if not Path(f).exists()]
    if missing_files:
        print(f"Error: Input files not found: {missing_files}")
        print("Run vote trace generation first to create the data files")
        return
    
    # Load signals (materialize list for reuse)
    signals_list = list(load_signals(args.inputs))
    if not signals_list:
        print("Error: No signals found in input files")
        return
    
    Path("tables").mkdir(exist_ok=True, parents=True)
    
    # Generate global top contributors table
    N, rows = aggregate_top_contrib(signals_list)
    render_jinja(Path(args.tpldir), "vote_contrib_table.tex.j2", 
                {"N": N, "rows": rows}, Path(args.out))
    
    # Generate SNR-stratified table if requested
    if args.snr_key and args.bins:
        try:
            bins = parse_bins(args.bins, pad_edges=args.pad_edges)
            bin_stats = aggregate_snr_top_contrib(signals_list, args.snr_key, bins, topk=args.topk_snr)
            render_jinja(Path(args.tpldir), "vt_top_contrib_snr.tex.j2",
                        {"bins": bin_stats, "snr_key": args.snr_key}, Path(args.out_snr))
            
            # Summary
            total_samples = sum(b["N"] for b in bin_stats)
            print(f"📊 SNR table summary:")
            print(f"   Total samples: {total_samples}")
            print(f"   Bins: {len(bin_stats)}")
            for b in bin_stats:
                if b["N"] > 0:
                    top_model = b["rows"][0]["model"] if b["rows"] else "None"
                    print(f"   {b['label']}: {b['N']} samples, top: {top_model}")
                    
        except Exception as e:
            print(f"Error generating SNR table: {e}")
            return
    
    # Final summary
    if rows:
        top_global = rows[0]
        print(f"📊 Global table summary:")
        print(f"   Samples: {N}")
        print(f"   Models: {len(rows)}")
        print(f"   Best contributor: {top_global['model']} (Δp = {top_global['mean_dp']:.4f})")


if __name__ == "__main__":
    main()