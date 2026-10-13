#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.ai_model_invoke import process_ai_predict_rainfall


def default_model_type() -> str:
    return "4"


def default_payload() -> dict[str, Any]:
    return {
        "event_id": "auto-20260414160000-1776155301995",
        "file_time": 1776057600,
        "satellite": {
            "type": "satellite",
            "event_id": "auto-20260414160000-1776155301995",
            "success": True,
            "data": "test_data/sat/clip",
        },
        "radar": {
            "type": "radar",
            "event_id": "auto-20260414160000-1776155301995",
            "success": True,
            "data": "test_data/achn",
        },
        "rain": {
            "type": "rain",
            "event_id": "auto-20260414160000-1776155301995",
            "success": True,
            "data": "test_data/rain",
        },
    }


def load_payload(payload_file: str) -> dict[str, Any]:
    if not payload_file:
        return default_payload()
    payload_path = Path(payload_file).expanduser()
    with payload_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_output_path(event_id: str, suffix: str) -> Path:
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_event = event_id.replace("/", "_")
    return out_dir / f"{timestamp}_{safe_event}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="测试 process_ai_predict_rainfall 调用")
    parser.add_argument("--payload-file", type=str, default="", help="可选：JSON 入参文件")
    parser.add_argument("--model-type", type=str, default="", help="可选：模型类型 1~7")
    args = parser.parse_args()

    payload = load_payload(args.payload_file)
    event_id = payload["event_id"]
    file_time = payload["file_time"]
    model_type = args.model_type or default_model_type()

    processed_data = {
        "satellite": payload.get("satellite", {}),
        "radar": payload.get("radar", {}),
        "rain": payload.get("rain", {}),
    }

    try:
        result = process_ai_predict_rainfall(
            event_id=event_id,
            file_time=file_time,
            model_type=model_type,
            processed_data=processed_data,
        )
        out_path = build_output_path(event_id, "json")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"Result JSON: {out_path}")
        print(f"Used times: {result.get('used_times', [])}")
        print(f"Pred shape: {result.get('pred_shape', [])}")
        print(f"Pred distribution: {result.get('pred_distribution', [])}")
        print(f"Input stats: {result.get('input_stats', {})}")
    except Exception:
        log_path = build_output_path(event_id, "log")
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Inference failed. Traceback saved to: {log_path}")
        raise


if __name__ == "__main__":
    main()
