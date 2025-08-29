#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick comparison of FORMAL / LIVELY / MIXED description JSONs.
Usage:
    python compare_descriptions.py \
        --data-root "G:/共用雲端硬碟/CNElab_林佳誼_Gaze/B.GazeImage/01.data/image_text_descriptions"
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate


# ---------- Utilities -------------------------------------------------- #
KEYWORDS = {
    "single": re.compile(r"\bsingle[\s\-]?player\b", re.I),
    "competition": re.compile(r"\b(comp(etition)?|competitive)\b", re.I),
    "cooperation": re.compile(r"\b(coop(eration)?|cooperative)\b", re.I),
}


def guess_label(text: str) -> str:
    """Rule-based label extraction."""
    for lbl, pat in KEYWORDS.items():
        if pat.search(text):
            return lbl.capitalize() if lbl != "single" else "Single"
    # fallback：看是否有 'playerA / playerB'
    if re.search(r"playerA", text, re.I):
        return "Competition"  # 競爭或合作；亂猜
    return "Unknown"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(paths):
    for name, path in paths.items():
        data = load_json(path)
        df = pd.DataFrame(data)

        # true label（你寫在 class 裡面的）
        true_labels = []
        for s in df["class"]:
            m = re.search(r"\b(single|competitive?|cooperative?)\b", s, re.I)
            true_labels.append(
                {
                    "single": "Single",
                    "competitive": "Competition",
                    "competition": "Competition",
                    "cooperative": "Cooperation",
                }.get(m.group(1).lower() if m else "unknown", "Unknown")
            )

        # predicted label
        pred_labels = df["class"].apply(guess_label).tolist()

        print(f"\n{'='*16}  {name.upper()}  {'='*16}")
        print(f"Samples: {len(df)}")
        if "Unknown" in pred_labels:
            unknown_cnt = Counter(pred_labels)["Unknown"]
            print(f"⚠️  {unknown_cnt} samples → 無法用規則判斷，標為 Unknown\n")

        print(
            classification_report(
                true_labels,
                pred_labels,
                labels=["Single", "Competition", "Cooperation"],
                digits=2,
                zero_division=0,
            )
        )


# ---------- CLI -------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default=".",
        help="資料夾路徑，裡面放三個 *_description_json.json",
    )
    args = ap.parse_args()

    paths = dict(
        formal=os.path.join(args.data_root, "formal_description_json.json"),
        lively=os.path.join(args.data_root, "lively_description_json.json"),
        mixed=os.path.join(args.data_root, "mixed_description_json.json"),
    )

    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    evaluate(paths)
