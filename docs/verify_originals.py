"""Check that every protected original file still matches the SHA-256 recorded before the dashboard was added.

Run from the project root:  python docs/verify_originals.py
"""
import hashlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
bad = missing = ok = 0
for line in (root / "docs" / "original_sha256_baseline.txt").read_text().splitlines():
    if not line.strip():
        continue
    digest, rel = line.split(None, 1)
    rel = rel.strip().lstrip("*")
    path = root / rel
    if not path.exists():
        print("MISSING ", rel); missing += 1
    elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        print("CHANGED ", rel); bad += 1
    else:
        ok += 1
print(f"{ok} identical, {bad} changed, {missing} missing")
sys.exit(1 if bad or missing else 0)
