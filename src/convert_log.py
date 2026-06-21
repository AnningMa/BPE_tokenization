#!/usr/bin/env python3
"""
convert_log.py — 把 tokenize-log.jsonl 展平成可读 CSV
用法:
    python convert_log.py                        # 只输出 CSV
    python convert_log.py --jsonl                # 同时输出格式化 JSONL
    python convert_log.py -i myfile.jsonl -o out # 自定义输入/输出路径
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_record(rec: dict) -> dict:
    """把一条嵌套 JSON 记录展平为单层 dict，列名用 . 分隔。"""
    flat = {}

    # 顶层字段
    for k in ("type", "train_data", "vocab_size", "min_pair_freq", "on_rare_words"):
        flat[k] = rec.get(k)

    # against_gold 子块
    ag = rec.get("against_gold", {})
    for k, v in ag.items():
        flat[f"gold.{k}"] = round(v, 6) if isinstance(v, float) else v

    # on_freq_words 子块
    fw = rec.get("on_freq_words", {})
    for k, v in fw.items():
        flat[f"freq.{k}"] = v

    # avg_fert_per_wt 子块
    ft = rec.get("avg_fert_per_wt", {})
    flat["fert.in_domain"] = ft.get("in_domain")
    flat["fert.out_domain"] = ft.get("out_domain")

    return flat


def main():
    parser = argparse.ArgumentParser(description="Convert tokenize-log.jsonl → CSV")
    parser.add_argument(
        "-i", "--input", default="../log/tokenize-log.jsonl", help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../log/tokenize-log",
        help="输出文件基名（不含扩展名）",
    )
    parser.add_argument(
        "--jsonl", action="store_true", help="同时输出格式化 JSONL（pretty-print）"
    )
    args = parser.parse_args()

    src = Path(args.input)
    records = [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # ── CSV ──────────────────────────────────────────────────────────────
    rows = [flatten_record(r) for r in records]
    df = pd.DataFrame(rows)

    # 数值列保留 6 位小数
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(6)

    csv_path = Path(args.output).with_suffix(".csv")
    df.to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )  # utf-8-sig 让 Excel 直接识别
    print(f"✓ CSV 已写入: {csv_path}  ({len(df)} 行 × {len(df.columns)} 列)")
    print()
    print(df.to_string(index=False))

    # ── 格式化 JSONL（可选）──────────────────────────────────────────────
    if args.jsonl:
        jsonl_path = Path(args.output + "_pretty").with_suffix(".jsonl")
        lines = [json.dumps(r, ensure_ascii=False, indent=2) for r in records]
        # 每条记录之间加一个空行，视觉上更易读
        jsonl_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n✓ 格式化 JSONL 已写入: {jsonl_path}")


if __name__ == "__main__":
    main()
