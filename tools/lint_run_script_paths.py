from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


"""Check that a run script only references paths that can exist.

Scripts for a new dataset are often derived from an existing one by
substitution, and a substitution that is too broad silently rewrites
paths that were already correct. That failure only shows up on the
server, one round trip per mistake, so it is checked here instead.

Repository-relative paths (tests/, tools/, configs/, manifests/,
scripts/, docs/) must exist locally. Absolute data and output paths
cannot be verified from here, so they are printed for review together
with any leftover token from the script this one was derived from.
"""


REPO_PREFIXES = (
    "tests/",
    "tools/",
    "configs/",
    "scripts/",
    "docs/",
)
PATH_PATTERN = re.compile(
    r"(?<![\w/.-])((?:tests|tools|configs|scripts|docs)/[\w./-]+)"
)
ABSOLUTE_PATTERN = re.compile(r"(/root/[\w./-]+)")
MODULE_PATTERN = re.compile(r"-m\s+(tools\.[\w.]+)")


def check(
    script: Path,
    *,
    project_root: Path,
    forbidden_tokens: tuple[str, ...],
) -> list[str]:
    text = script.read_text(encoding="utf-8")
    problems: list[str] = []

    if "\r\n" in text:
        problems.append("contains CRLF line endings")

    for relative in sorted(set(PATH_PATTERN.findall(text))):
        if relative.endswith((".py", ".yaml", ".yml", ".sh", ".md", ".csv")):
            if not (project_root / relative).is_file():
                problems.append(f"missing repository file: {relative}")

    for module in sorted(set(MODULE_PATTERN.findall(text))):
        candidate = project_root / (module.replace(".", "/") + ".py")
        if not candidate.is_file():
            problems.append(f"missing module: {module}")

    for token in forbidden_tokens:
        for number, line in enumerate(text.splitlines(), start=1):
            if token in line and "tests/" not in line:
                problems.append(
                    f"line {number}: leftover token {token!r}: {line.strip()}"
                )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a run script references only existing "
            "repository files and no leftover tokens from the script it "
            "was derived from."
        )
    )
    parser.add_argument("scripts", nargs="+")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--forbidden-token",
        action="append",
        default=[],
        help="Token that must not survive the derivation, e.g. dsifn",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    failed = False
    for name in args.scripts:
        script = Path(name)
        problems = check(
            script,
            project_root=project_root,
            forbidden_tokens=tuple(args.forbidden_token),
        )
        text = script.read_text(encoding="utf-8")
        print(f"=== {name}")
        for absolute in sorted(set(ABSOLUTE_PATTERN.findall(text))):
            print(f"    absolute path (review): {absolute}")
        if problems:
            failed = True
            for problem in problems:
                print(f"    PROBLEM: {problem}")
        else:
            print("    ok")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
