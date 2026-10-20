from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


"""Check whether label morphology alone predicts recoverable headroom.

The applicability protocol claims that a dataset can be screened for task
utility before a detector is trained, because the recoverable loss under
a fixed pixel displacement depends on how large the changed objects are
relative to that displacement. So far that claim rests on two datasets.

This tool tests it against every dataset for which both quantities are
already measured: the label-only morphology audit on one side, and the
input-oracle headroom measured with a trained detector on the other. It
computes no new metric and trains nothing; it checks whether the cheap
signal orders the datasets the same way the expensive one does.

A rank agreement is evidence that the screen is usable. A disagreement
would mean the screen must not be presented as predictive, and the
protocol would have to fall back on the decisive headroom gate alone.
"""


def spearman_footrule(
    proxy: list[float],
    measured: list[float],
) -> dict[str, Any]:
    """Rank agreement without a scipy dependency.

    Returns Spearman's rho computed from ranks, plus the concordant pair
    fraction, which is easier to state honestly for four points than a
    correlation coefficient.
    """
    if len(proxy) != len(measured):
        raise ValueError("Proxy and measured series must be the same length")
    if len(proxy) < 3:
        raise ValueError("Rank agreement needs at least three datasets")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while (
                stop + 1 < len(order)
                and values[order[stop + 1]] == values[order[index]]
            ):
                stop += 1
            average = (index + stop) / 2.0 + 1.0
            for position in range(index, stop + 1):
                result[order[position]] = average
            index = stop + 1
        return result

    proxy_ranks = ranks(proxy)
    measured_ranks = ranks(measured)
    count = len(proxy)
    mean_proxy = sum(proxy_ranks) / count
    mean_measured = sum(measured_ranks) / count
    covariance = sum(
        (p - mean_proxy) * (m - mean_measured)
        for p, m in zip(proxy_ranks, measured_ranks)
    )
    proxy_variance = sum((p - mean_proxy) ** 2 for p in proxy_ranks)
    measured_variance = sum(
        (m - mean_measured) ** 2 for m in measured_ranks
    )
    denominator = (proxy_variance * measured_variance) ** 0.5
    rho = covariance / denominator if denominator else float("nan")

    concordant = 0
    discordant = 0
    for i in range(count):
        for j in range(i + 1, count):
            proxy_delta = proxy[i] - proxy[j]
            measured_delta = measured[i] - measured[j]
            if proxy_delta == 0 or measured_delta == 0:
                continue
            if (proxy_delta > 0) == (measured_delta > 0):
                concordant += 1
            else:
                discordant += 1
    pairs = concordant + discordant
    return {
        "spearman_rho": rho,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "concordant_fraction": concordant / pairs if pairs else float("nan"),
        "datasets": count,
    }


def build_validation(
    records: list[dict[str, Any]],
    *,
    headroom_gate: float,
) -> dict[str, Any]:
    """Compare the label-only proxy with measured oracle headroom.

    The proxy is 1 - self-IoU of the label under the same displacement:
    a mask that barely moves against itself leaves little for alignment
    to recover, so a larger value should accompany larger headroom.
    """
    for record in records:
        record["proxy_displaced_label_disagreement"] = 1.0 - float(
            record["mask_self_iou_shift8"]
        )
    ordered = sorted(
        records,
        key=lambda item: item["proxy_displaced_label_disagreement"],
    )
    agreement = spearman_footrule(
        [
            float(item["proxy_displaced_label_disagreement"])
            for item in ordered
        ],
        [float(item["measured_shift8_headroom"]) for item in ordered],
    )
    separates_gate = all(
        (
            float(a["measured_shift8_headroom"]) >= headroom_gate
        ) == (
            a["proxy_displaced_label_disagreement"]
            >= min(
                item["proxy_displaced_label_disagreement"]
                for item in records
                if float(item["measured_shift8_headroom"])
                >= headroom_gate
            )
        )
        for a in records
    )
    return {
        "status": "completed",
        "analysis_type": "proxy_validation_no_training",
        "test_split_touched": False,
        "headroom_gate": headroom_gate,
        "datasets": ordered,
        "rank_agreement": agreement,
        "proxy_separates_gate_decisions": separates_gate,
        "interpretation": (
            "Rank agreement over a handful of datasets is suggestive, not "
            "conclusive. It supports using the label-only screen to order "
            "candidates before training, and does not replace the "
            "decisive headroom gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether label-only morphology predicts measured "
            "input-oracle headroom across datasets."
        )
    )
    parser.add_argument(
        "--records",
        required=True,
        help=(
            "JSON list with dataset, mask_self_iou_shift8 and "
            "measured_shift8_headroom per dataset."
        ),
    )
    parser.add_argument("--headroom-gate", type=float, default=0.05)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    report = build_validation(
        records, headroom_gate=float(args.headroom_gate)
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for item in report["datasets"]:
        print(
            f"{item['dataset']:>12s}  proxy "
            f"{item['proxy_displaced_label_disagreement']:.3f}  "
            f"headroom {float(item['measured_shift8_headroom']):.4f}"
        )
    print(json.dumps(report["rank_agreement"], indent=2))


if __name__ == "__main__":
    main()
