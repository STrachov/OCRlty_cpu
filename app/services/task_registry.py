from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException


APP_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = APP_ROOT / "schemas"
PROMPTS_DIR = APP_ROOT / "prompts"

TASK_SPECS = [
    {
        "task_id": "receipt_fields_v1",
        "prompt_file": "receipt_fields_v1.txt",
        "schema_file": "receipt_fields_v1.schema.json",
        "description": "Extracts receipt-level monetary totals and payment fields as raw printed strings.",
    },
    {
        "task_id": "items_light_v1",
        "prompt_file": "items_light_v1.txt",
        "schema_file": "items_light_v1.schema.json",
        "description": "Extracts purchased line items with raw quantity, unit price, and line total values.",
    },
]

_TASKS: Dict[str, Dict[str, Any]] = {}
_TASK_LOAD_ERRORS: List[str] = []
_LOG = logging.getLogger("ocrlty")


def load_tasks() -> None:
    global _TASKS, _TASK_LOAD_ERRORS
    _TASKS = {}
    _TASK_LOAD_ERRORS = []

    for spec in TASK_SPECS:
        task_id = spec["task_id"]
        try:
            prompt_path = (PROMPTS_DIR / spec["prompt_file"]).resolve()
            schema_path = (SCHEMAS_DIR / spec["schema_file"]).resolve()
            prompt_value = prompt_path.read_text(encoding="utf-8")
            schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
            if not isinstance(schema_value, dict):
                raise ValueError("schema JSON must be an object")
            _TASKS[task_id] = {
                "task_id": task_id,
                "description": str(spec.get("description") or ""),
                "prompt_path": str(prompt_path),
                "schema_path": str(schema_path),
                "prompt_value": prompt_value,
                "schema_value": schema_value,
            }
        except Exception as e:
            _TASK_LOAD_ERRORS.append(f"{task_id}: {e}")


def get_task(task_id: str) -> Dict[str, Any]:
    if not _TASKS:
        load_tasks()
    task = _TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id!r}. Available: {sorted(_TASKS.keys())}")
    if _TASK_LOAD_ERRORS:
        _LOG.warning("task_load_errors", extra={"event": "task_load_errors", "errors": _TASK_LOAD_ERRORS})
    return task


def list_task_summaries() -> List[Dict[str, str]]:
    if not _TASKS:
        load_tasks()
    return [
        {
            "task_id": task_id,
            "description": str((_TASKS.get(task_id) or {}).get("description") or ""),
        }
        for task_id in sorted(_TASKS.keys())
    ]
