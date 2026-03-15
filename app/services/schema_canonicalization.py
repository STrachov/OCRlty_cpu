from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple

from app.settings import settings


def canon_x_string(v: Any, *, str_mode: str = "exact") -> Tuple[bool, str, Optional[str]]:
    if v is None:
        return True, "", None
    s = str(v)
    if str_mode == "strip":
        s = s.strip()
    elif str_mode == "strip_collapse":
        s = " ".join(s.strip().split())
    return True, s, None


def get_decimal_sep() -> str:
    return str(getattr(settings, "DECIMAL_SEP", ".") or ".").strip()[:1] or "."


def _normalize_grouped_integer(integer_part: str, *, group_sep: str) -> Tuple[bool, Optional[str], Optional[str]]:
    if not integer_part:
        return True, "0", None
    if group_sep not in integer_part:
        return (True, integer_part, None) if integer_part.isdigit() else (False, None, "invalid digits")

    groups = integer_part.split(group_sep)
    if not groups or any(not g for g in groups):
        return False, None, "invalid grouping"
    if not groups[0].isdigit() or len(groups[0]) > 3:
        return False, None, "invalid grouping"
    for g in groups[1:]:
        if not g.isdigit() or len(g) != 3:
            return False, None, "invalid grouping"
    return True, "".join(groups), None


def _canon_numeric_string(
    v: Any,
    *,
    decimal_sep: Optional[str],
    allow_fraction: bool,
    label: str,
) -> Tuple[bool, Optional[str], Optional[str]]:
    if v is None:
        return True, None, None
    s = str(v).strip()
    if not s:
        return True, None, None

    dec = (decimal_sep or get_decimal_sep()).strip()[:1] or "."
    grp = "," if dec == "." else "."
    s = s.replace(" ", "").replace("\u00a0", "")
    s = re.sub(r"[^0-9,\.\-]+", "", s)
    if s in {"", "-", dec, f"-{dec}"}:
        return False, None, f"{label}: cannot parse"

    negative = s.startswith("-")
    body = s[1:] if negative else s
    if "-" in body:
        return False, None, f"{label}: cannot parse"
    if body.count(dec) > 1:
        return False, None, f"{label}: cannot parse"

    if dec in body:
        integer_part, fraction_part = body.split(dec, 1)
    else:
        integer_part, fraction_part = body, ""

    if grp in fraction_part:
        return False, None, f"{label}: invalid grouping"

    ok_int, normalized_int, int_err = _normalize_grouped_integer(integer_part, group_sep=grp)
    if not ok_int or normalized_int is None:
        return False, None, f"{label}: {int_err or 'cannot parse'}"

    if fraction_part and not fraction_part.isdigit():
        return False, None, f"{label}: cannot parse"

    if not allow_fraction:
        if fraction_part.strip("0"):
            return False, None, f"{label}: fractional part is not allowed"
        fraction_part = ""

    normalized_int = normalized_int.lstrip("0") or "0"
    normalized = normalized_int
    if fraction_part:
        normalized_fraction = fraction_part.rstrip("0")
        if normalized_fraction:
            normalized = f"{normalized_int}.{normalized_fraction}"

    if negative and normalized != "0":
        normalized = f"-{normalized}"

    try:
        Decimal(normalized)
    except InvalidOperation:
        return False, None, f"{label}: cannot parse"
    return True, normalized, None


def canon_x_money(v: Any, *, decimal_sep: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    return _canon_numeric_string(v, decimal_sep=decimal_sep, allow_fraction=True, label="x-money")


def canon_x_count(v: Any, *, decimal_sep: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    return _canon_numeric_string(v, decimal_sep=decimal_sep, allow_fraction=False, label="x-count")


def _deref_schema(schema: Dict[str, Any], *, root_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    parts = ref[2:].split("/")
    cur: Any = root_schema
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return schema
        cur = cur[p]
    return cur if isinstance(cur, dict) else schema


def canonicalize_value_for_schema(value: Any, schema_node: Dict[str, Any], *, root_schema: Dict[str, Any], str_mode: str = "exact") -> Any:
    schema_node = _deref_schema(schema_node, root_schema=root_schema)
    t = schema_node.get("type")
    if value is None:
        return None

    if t == "object":
        if not isinstance(value, dict):
            return value
        props = schema_node.get("properties") if isinstance(schema_node.get("properties"), dict) else {}
        out: Dict[str, Any] = {}
        for key, raw in value.items():
            child_schema = props.get(key) if isinstance(props.get(key), dict) else None
            out[key] = canonicalize_value_for_schema(raw, child_schema or {}, root_schema=root_schema, str_mode=str_mode)
        return out

    if t == "array":
        if not isinstance(value, list):
            return value
        items_schema = schema_node.get("items") if isinstance(schema_node.get("items"), dict) else {}
        return [
            canonicalize_value_for_schema(item, items_schema, root_schema=root_schema, str_mode=str_mode)
            for item in value
        ]

    fmt = schema_node.get("format")
    if fmt == "x-money":
        ok, canon, _ = canon_x_money(value)
        return canon if ok else value
    if fmt == "x-count":
        ok, canon, _ = canon_x_count(value)
        return canon if ok else value

    ok, canon, _ = canon_x_string(value, str_mode=str_mode)
    return canon if ok else value
