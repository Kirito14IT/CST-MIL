"""Capture/verify protected inputs without modifying any protected file."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("protected_files.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths():
    files = [ROOT / name for name in ["任务要求.docx", "任务要求 - 副本.docx",
                                      "任务进度报告.md", "任务进度报告.pdf"]]
    for folder in ["cst_mil/data/raw", "cst_mil/data/prepared", "cst_mil/artifacts/run_001"]:
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    return sorted(files)


if __name__ == "__main__":
    actual = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths()}
    if MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        bad = [p for p in set(expected) | set(actual) if expected.get(p) != actual.get(p)]
        print(json.dumps({"protected_files": len(actual), "mismatches": bad}, ensure_ascii=True))
        sys.exit(bool(bad))
    MANIFEST.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Captured {len(actual)} protected files")
