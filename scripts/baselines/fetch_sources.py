import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "third_party/baselines/UPSTREAMS.json").read_text(encoding="utf-8-sig"))
    records = {item["name"]: item for item in manifest["repositories"]}
    names = args.names or list(records)
    unknown = set(names) - records.keys()
    if unknown:
        parser.error(f"Unknown repositories: {sorted(unknown)}")
    for name in names:
        record = records[name]
        destination = (root / record["path"]).resolve()
        if not destination.is_relative_to(root / "third_party/baselines"):
            raise ValueError(f"Unexpected repository path: {destination}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--no-checkout", "--depth", "1", record["url"], str(destination)], check=True
            )
            subprocess.run(
                ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", record["commit"]], check=True
            )
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", record["commit"]], check=True)
        remote = subprocess.check_output(
            ["git", "-C", str(destination), "remote", "get-url", "origin"], text=True
        ).strip()
        commit = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True).strip()
        if remote != record["url"] or commit != record["commit"] or dirty:
            raise RuntimeError(f"{name} is not the pinned clean checkout; left unchanged: {destination}")
        print(f"{name}: {commit}", flush=True)


if __name__ == "__main__":
    main()
