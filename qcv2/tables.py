"""LaTeX table bodies (numbers only, no captions). Writes results/v2/tables/*.tex."""
from __future__ import annotations

import numpy as np

from .paths import TEX


def _agg(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "n/a", 0
    return f"{np.mean(vals):.3f}", len(vals)


def table_from_rows(rows, metrics_list, name, head_col="head", with_collapsed=True):
    """Generic mean table: rows indexed by (task, head), columns = metrics."""
    TEX.mkdir(parents=True, exist_ok=True)
    keyed = {}
    for r in rows:
        if r.get("collapsed") == "True" and not with_collapsed:
            continue
        k = (r.get("task", ""), r.get(head_col, ""))
        keyed.setdefault(k, {}).setdefault(r["metric"], []).append(float(r["estimate"]))
    lines = ["\\begin{tabular}{ll" + "c" * len(metrics_list) + "}",
             "\\toprule",
             "Task & Head & " + " & ".join(metrics_list) + " \\\\",
             "\\midrule"]
    for (task, head) in sorted(keyed):
        cells = []
        for m in metrics_list:
            val, n = _agg(keyed[(task, head)].get(m, []))
            cells.append(f"{val} ({n})")
        lines.append(f"{task.replace('_', ' ')} & {head.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    suffix = "" if with_collapsed else "_nocollapsed"
    (TEX / f"{name}{suffix}.tex").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)