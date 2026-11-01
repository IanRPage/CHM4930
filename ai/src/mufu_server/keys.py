"""
Named API keys, stored as sha256 hashes in a file of `<name> <hash>` lines. A name can
be an email address; it only labels the key and isn't verified.

From ai/: PYTHONPATH=src python -m mufu_server.keys add <name> [--file PATH]
prints a new key once and saves its hash. Delete a key's line to revoke it; changes
apply when the server restarts.
"""

import argparse
import hashlib
import os
import re
import secrets
import sys
from pathlib import Path

DEFAULT_FILE = Path.home() / ".config" / "mufu" / "api_keys"
NAME = re.compile(r"[A-Za-z0-9_.+@-]+")
HASH = re.compile(r"[0-9a-f]{64}")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_keys(path: Path) -> dict[str, str]:
    """
    Name to key hash. Raises if others can write the file, on malformed lines, or on
    repeated names or keys.
    """
    # only hashes are stored, so being readable is fine, but a writer could add a key
    if path.stat().st_mode & 0o022:
        raise ValueError(f"{path} is writable by other users; chmod 600 it")
    keys = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if (
            len(parts) != 2
            or not NAME.fullmatch(parts[0])
            or not HASH.fullmatch(parts[1])
        ):
            raise ValueError(f"{path}:{n}: expected '<name> <sha256>'")
        if parts[0] in keys:
            raise ValueError(f"{path}:{n}: {parts[0]} is listed twice")
        if parts[1] in keys.values():
            raise ValueError(f"{path}:{n}: {parts[0]} reuses another name's key")
        keys[parts[0]] = parts[1]
    return keys


def add_key(path: Path, name: str) -> str:
    """
    Append a new key for `name` (file created owner-only) and return the key.
    """
    if not NAME.fullmatch(name):
        raise ValueError("names may only use letters, digits and _ . + @ -")
    text = path.read_text() if path.exists() else ""
    if text and name in load_keys(path):
        raise ValueError(f"{name} already has a key; delete its line to replace it")
    key = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        # a hand-edited file may not end in a newline
        sep = "\n" if text and not text.endswith("\n") else ""
        f.write(f"{sep}{name} {hash_key(key)}\n")
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mufu_server.keys")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="create a key and print it once")
    add.add_argument("name")
    add.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args(argv)
    try:
        key = add_key(args.file, args.name)
    except ValueError as e:
        parser.error(str(e))
    print(key)
    print(f"saved {args.name}'s hash to {args.file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
