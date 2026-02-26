# app/auth_cli.py
import argparse
import json
import sys
from typing import List, Optional

from app.settings import settings
from app.auth import (
    ROLE_PRESET_SCOPES,
    create_api_key,
    init_auth_db,
    list_api_keys,
    revoke_api_key,
)


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def _parse_scopes(scopes_arg: Optional[str]) -> Optional[List[str]]:
    if not scopes_arg:
        return None
    s = scopes_arg.strip()
    if not s:
        return None
    # Allow JSON array or CSV
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        prog="auth-cli",
        description="Local CLI for managing API keys (PostgreSQL).",
    )
    p.add_argument(
        "--database-url",
        "--db",
        default=settings.DATABASE_URL,
        help=f"PostgreSQL DATABASE_URL (default: {settings.DATABASE_URL})",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sub_init = sub.add_parser("init-db", help="Validate DB connection/schema")
    sub_init.set_defaults(func=lambda args: init_auth_db(args.database_url))


    sub_create = sub.add_parser("create-key", help="Create a new API key")
    sub_create.add_argument("--key-id", required=True, help="Human-readable key id (unique)")
    sub_create.add_argument("--role", required=True, choices=sorted(ROLE_PRESET_SCOPES.keys()))
    sub_create.add_argument(
        "--scopes",
        default=None,
        help="Optional scopes override. CSV 'a,b,c' or JSON array '[a,b]'. "
             "If omitted, role preset scopes are used.",
    )

    def _create(args):
        if not settings.API_KEY_PEPPER:
            _die("API_KEY_PEPPER is not set. Set env var before creating keys.", 2)
        scopes = _parse_scopes(args.scopes)
        raw_key, principal = create_api_key(
            key_id=args.key_id,
            role=args.role,
            scopes=scopes,
            database_url=args.database_url,
        )
        print("=== API KEY CREATED (shown only once) ===")
        print(f"key_id:  {principal.key_id}")
        print(f"role:    {principal.role}")
        print(f"scopes:  {sorted(list(principal.scopes))}")
        print(f"api_key: {raw_key}")
        print("=======================================")

    sub_create.set_defaults(func=_create)

    sub_revoke = sub.add_parser("revoke-key", help="Revoke (deactivate) a key by key_id")
    sub_revoke.add_argument("--key-id", required=True)

    def _revoke(args):
        ok = revoke_api_key(args.key_id, database_url=args.database_url)
        if ok:
            print(f"Revoked: {args.key_id}")
        else:
            print(f"Not found or already inactive: {args.key_id}")

    sub_revoke.set_defaults(func=_revoke)

    sub_list = sub.add_parser("list-keys", help="List keys (no secrets)")

    def _list(args):
        rows = list_api_keys(database_url=args.database_url)
        if not rows:
            print("(no keys)")
            return
        for r in rows:
            print(
                f"[{r['api_key_id']:04d}] "
                f"key_id={r['key_id']} role={r['role']} active={r['is_active']} "
                f"created_at={r['created_at']} last_used_at={r['last_used_at']} revoked_at={r['revoked_at']} "
                f"scopes={r['scopes']}"
            )

    sub_list.set_defaults(func=_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
