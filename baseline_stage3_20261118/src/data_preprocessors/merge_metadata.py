"""
Merge metadata.parquet files from subdirectories.

This script merges all metadata.parquet files found in subdirectories
of a specified root directory into a single parquet file.
"""

import argparse
from pathlib import Path

import natsort
import pandas as pd

from src.utils.logging import log_print


def merge_metadata_files(root_dir, output_file, add_source_col=True):
    """
    Merge all metadata.parquet files from subdirectories.

    Args:
        root_dir (str or Path): Root directory containing subdirectories with metadata.parquet files.
        output_file (str or Path): Output path for the merged parquet file.
        add_source_col (bool): Whether to add a column identifying the source subdirectory.
    """
    root_path = Path(root_dir)
    all_metadata = []
    dirs = [f for f in root_path.iterdir() if f.is_dir()]
    dirs = natsort.natsorted(dirs, key=lambda x: x.name)

    log_print(f"Dirs are {dirs}")

    # Iterate through all subdirectories in the root directory
    for subdir in dirs:
        if subdir.is_dir():
            metadata_file = subdir / "metadata.parquet"
            if metadata_file.exists():
                try:
                    # Read the parquet file
                    df = pd.read_parquet(metadata_file)

                    # Optionally add a column to identify the source subdirectory
                    if add_source_col:
                        df["source_subdir"] = subdir.name

                    # Add to the list
                    all_metadata.append(df)
                    log_print(f"Loaded {len(df)} records from {metadata_file}")
                except Exception as e:
                    log_print(f"Error reading {metadata_file}: {e}", "error")

    if not all_metadata:
        log_print("No metadata files found.")
        return

    # Concatenate all DataFrames
    merged_df = pd.concat(all_metadata, ignore_index=True)
    log_print(f"Total records after merging: {len(merged_df)}")

    # Reset index
    merged_df.reset_index(drop=True, inplace=True)

    # Save the merged DataFrame to a new parquet file
    merged_df.to_parquet(output_file)
    log_print(f"Merged metadata saved to {output_file}")

    # Print basic information
    log_print("\nMerged DataFrame info:")
    log_print(f"Shape: {merged_df.shape}")
    log_print(f"Columns: {list(merged_df.columns)}")

    # Show first few rows
    log_print("\nFirst few rows:")
    log_print(merged_df.head().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Merge metadata.parquet files from subdirectories"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data_original/zihan_processed/wds_interval_30",
        help="Root directory containing subdirectories with metadata.parquet files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data_original/zihan_processed/wds_interval_30/train_test/test_metadata.parquet",
        help="Output file path for merged metadata",
    )
    parser.add_argument(
        "--no_source_col",
        action="store_true",
        help="Don't add source_subdir column to identify origin",
    )

    args = parser.parse_args()

    merge_metadata_files(
        root_dir=args.root_dir,
        output_file=args.output_file,
        add_source_col=not args.no_source_col,
    )


if __name__ == "__main__":
    main()
