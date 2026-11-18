from __future__ import annotations

import argparse

from src.tools.rain_station_excel_to_shard_db import ShardedRainDataImporter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="data2/rainfall_shards")
    ap.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    ds = ShardedRainDataImporter(args.base_dir)
    rel = ds.get_relevant_shards(args.date, args.date)
    print(f"base_dir={args.base_dir}")
    print(f"date={args.date}")
    print(f"n_relevant_shards={len(rel)}")
    for s in rel:
        print(s)


if __name__ == "__main__":
    main()

