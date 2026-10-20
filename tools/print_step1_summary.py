"""Print a step 1 decision (P25 / P26 / P27 harness) as a compact table."""
from __future__ import annotations

import json
import sys


def fmt(value, digits=3) -> str:
    return "None" if value is None else f"{value:.{digits}f}"


def render(decision: dict) -> str:
    lines = [f"verdicts: {decision['verdicts']}"]
    for name, entry in decision["datasets"].items():
        lines.append(f"\n[{name}] verdict {entry['verdict']}")
        if "levels" not in entry:
            continue
        lines.append(f"  crops {entry['crops']} from {entry['tiles']} tiles; clean trigger {fmt(entry['clean_trigger_rate'])}")
        lines.append("  level  success  converged  trigger  perturb_px  ate_raw  ate_diff  sec")
        for level, v in entry["levels"].items():
            lines.append(
                f"  {level:>5}  {fmt(v.get('success_rate')):>7}  {fmt(v['converged_rate']):>9}  "
                f"{fmt(v['trigger_rate']):>7}  {fmt(v['median_perturbation_px'], 1):>10}  "
                f"{fmt(v['median_ate_raw'], 2):>7}  {fmt(v['median_ate_diff'], 2):>8}  {fmt(v['median_seconds'], 3)}"
            )
        lines.append(f"  clean median displacement px: {fmt(entry['levels']['0'].get('median_clean_displacement_px'), 2)}")
        lines.append("  change bins (crops / top-level success / clean trigger): " + " | ".join(
            f"{b['range'][0]:.2f}-{b['range'][1]:.2f}: {b['crops']} / {fmt(b['top_level_success'], 2)} / "
            f"{fmt(b['clean_trigger_rate'], 2)}" for b in entry["change_bins"]
        ))
    return "\n".join(lines)


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as handle:
        print(render(json.load(handle)))


if __name__ == "__main__":
    main()
