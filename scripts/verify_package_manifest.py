"""Verify every packaged file against metadata/MANIFEST.sha256.

Transient artefacts are ignored: interpreter and test-runner caches
(``__pycache__``, ``.pytest_cache``, ``.mypy_cache``, ``.ipynb_checkpoints``)
are produced by running the verification steps themselves, and ``.git`` appears
when the artifact is distributed as a repository. All three are outside the
distributed file set, so ignoring them keeps this check usable both from the
released archive and from a fresh clone.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "MANIFEST.sha256"
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
                ".ruff_cache", ".git", ".idea", ".vscode", ".DS_Store"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_transient(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)


def main() -> None:
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    actual_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST and not is_transient(path)
    }
    missing = sorted(set(expected) - actual_files)
    extra = sorted(actual_files - set(expected))
    mismatched = sorted(
        relative
        for relative, digest in expected.items()
        if (ROOT / relative).exists() and sha256(ROOT / relative) != digest
    )
    if missing or extra or mismatched:
        raise SystemExit(
            f"Manifest verification failed: missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    print(f"Manifest verified: {len(expected)} files")


if __name__ == "__main__":
    main()
