"""Pipeline step: generate logged AI commentary for a curve-view summary.

Example:
    .venv/bin/python pipeline_steps/06_generate_ai_commentary.py \
      --summary data/03_processed/germany_modelling_2021_2026/lear_model_lasso_raw/03_curve_translation/20251101_20251201/baseload/curve_view_summary.csv

The script calls an OpenAI model using ``OPENAI_API_KEY`` from ``.env`` or the
process environment. It logs the prompt, output, and failures next to the
generated commentary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIRequestError(RuntimeError):
    """Raised when the OpenAI API returns a non-success response."""


def parse_command_line_arguments() -> argparse.Namespace:
    """Read commentary generation settings from the command line."""

    parser = argparse.ArgumentParser(description="Generate AI commentary for a curve view.")
    parser.add_argument("--summary", required=True, help="Path to curve_view_summary.csv.")
    parser.add_argument("--env", default=".env", help="Path to local .env file.")
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Where to write commentary and logs. Defaults next to the summary CSV.",
    )
    return parser.parse_args()


def utc_timestamp_slug() -> str:
    """Return a filesystem-safe UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_curve_summary(path: str | Path) -> dict[str, object]:
    """Read the one-row curve-view summary as a dictionary."""

    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f"Curve-view summary is empty: {path}")
    return table.iloc[0].to_dict()


def build_prompt(summary: dict[str, object]) -> str:
    """Build a constrained prompt that forbids invented market facts."""

    summary_json = json.dumps(summary, indent=2, sort_keys=True, default=str)
    return f"""You are writing a concise power trading desk commentary.

Use only the numbers and fields in the JSON below. Do not invent outages,
weather, flow changes, market news, prices, dates, or model metrics. If a fact
is not present, say it is not available. Keep the wording practical and
decision-focused.

Explain:
1. The selected delivery period and block.
2. Forecast fair value and its P10-P90 band versus the benchmark.
3. The resulting long/short/neutral signal and how far the benchmark sits from the band.
4. How MAE and the provided tail metric frame the model's reliability.
5. What would invalidate the signal.

Return Markdown only.

Curve-view JSON:
```json
{summary_json}
```
"""


def write_json_log(path: Path, data: dict[str, object]) -> None:
    """Write one JSON log file."""

    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def call_openai_responses_api(prompt: str, api_key: str, model: str) -> dict[str, object]:
    """Call OpenAI's Responses API using direct HTTP."""

    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": prompt,
        },
        timeout=60,
    )
    if not response.ok:
        raise OpenAIRequestError(
            f"OpenAI API returned HTTP {response.status_code}: {response.text}"
        )
    return response.json()


def extract_output_text(response_json: dict[str, object]) -> str:
    """Extract text from a Responses API JSON object."""

    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # Fallback for response shapes where text appears inside output blocks.
    text_parts: list[str] = []
    for item in response_json.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                text_parts.append(content["text"])

    if not text_parts:
        raise ValueError("OpenAI response did not contain extractable output text.")
    return "\n".join(text_parts)


def deterministic_fallback_commentary(summary: dict[str, object], failure_reason: str) -> str:
    """Create a non-LLM commentary so the pipeline still completes."""

    return f"""# Fallback Curve Commentary

The LLM commentary call failed, so this deterministic fallback was generated from the curve-view summary only.

- Period UTC: `{summary.get("period_start_utc")}` to `{summary.get("period_end_utc")}`
- Block: `{summary.get("block")}`
- Signal: **{summary.get("signal")}**
- Forecast fair value: `{summary.get("forecast_fair_value")}` EUR/MWh
- Forecast band: `{summary.get("forecast_low")}` to `{summary.get("forecast_high")}` EUR/MWh (`{summary.get("band_source")}`)
- Benchmark method: `{summary.get("benchmark_method")}`
- Benchmark value: `{summary.get("benchmark_value")}` EUR/MWh
- Edge vs benchmark: `{summary.get("edge")}` EUR/MWh
- Distance beyond band edge: `{summary.get("signal_margin")}` EUR/MWh
- MAE: `{summary.get("mae")}` EUR/MWh
- Tail metric: `{summary.get("tail_metric_name")}` = `{summary.get("tail_metric_value")}` EUR/MWh
- Prediction coverage: `{summary.get("prediction_coverage")}`

Desk action: {summary.get("desk_action")}

Decision rationale: {summary.get("decision_rationale")}

Invalidation logic: {summary.get("invalidation_logic")}

LLM failure reason: `{failure_reason}`
"""


def main() -> None:
    """Generate commentary and required logs."""

    args = parse_command_line_arguments()
    load_dotenv(args.env)

    summary_path = Path(args.summary)
    output_folder = Path(args.output_folder) if args.output_folder else summary_path.parent
    logs_folder = output_folder / "ai_logs"
    logs_folder.mkdir(parents=True, exist_ok=True)
    timestamp = utc_timestamp_slug()
    failure_log_path = logs_folder / f"{timestamp}_failure.json"

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

    summary = read_curve_summary(summary_path)
    prompt = build_prompt(summary)
    prompt_log_path = logs_folder / f"{timestamp}_prompt.json"
    output_log_path = logs_folder / f"{timestamp}_output.json"
    commentary_path = output_folder / "ai_commentary.md"

    write_json_log(
        prompt_log_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "summary_path": str(summary_path),
            "prompt": prompt,
            "input_summary": summary,
        },
    )

    if not api_key:
        failure_reason = "OPENAI_API_KEY is missing from .env or the environment."
        commentary_path.write_text(
            deterministic_fallback_commentary(summary, failure_reason),
            encoding="utf-8",
        )
        write_json_log(
            failure_log_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "summary_path": str(summary_path),
                "commentary_path": str(commentary_path),
                "error_type": "MissingOpenAIAPIKey",
                "error_message": failure_reason,
                "fallback_used": True,
            },
        )
        print(f"Fallback commentary saved: {commentary_path}")
        print(f"Failure log saved: {failure_log_path}")
        return

    try:
        response_json = call_openai_responses_api(prompt, api_key, model)
        commentary = extract_output_text(response_json)
        commentary_path.write_text(commentary, encoding="utf-8")
        write_json_log(
            output_log_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "commentary_path": str(commentary_path),
                "response": response_json,
            },
        )
    except Exception as exc:
        commentary_path.write_text(
            deterministic_fallback_commentary(summary, str(exc)),
            encoding="utf-8",
        )
        write_json_log(
            failure_log_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "summary_path": str(summary_path),
                "commentary_path": str(commentary_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "fallback_used": True,
            },
        )
        print(f"Fallback commentary saved: {commentary_path}")
        print(f"Failure log saved: {failure_log_path}")
        return

    print(f"AI commentary saved: {commentary_path}")
    print(f"Prompt log saved: {prompt_log_path}")
    print(f"Output log saved: {output_log_path}")


if __name__ == "__main__":
    main()
