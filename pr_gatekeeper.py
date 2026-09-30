"""
pr_gatekeeper.py — two-speed code review with Jev + an LLM.

System One (Jev): triages every pull request diff in one call — area, risk,
security impact, breaking changes, test coverage — and returns calibrated
probabilities instead of text.

System Two (LLM): only runs when Jev flags the PR as risky OR isn't confident.
Most PRs (docs, small fixes) never pay for a full LLM review.

Usage:
    pip install langchain-typesafe anthropic
    export TYPESAFE_API_KEY=...  ANTHROPIC_API_KEY=...
    python pr_gatekeeper.py                 # diffs HEAD against origin/main
    python pr_gatekeeper.py path/to/pr.diff # or review a saved diff
"""

import os
import subprocess
import sys
import time

import anthropic
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

REVIEW_MODEL = os.getenv("REVIEW_MODEL", "claude-opus-4-8")
BASE_BRANCH = os.getenv("BASE_BRANCH", "origin/master")
MAX_DIFF_CHARS = 20_000

FLAG_THRESHOLD = 0.5      # Noul probability that counts as "yes"
MIN_CONFIDENCE = 0.6      # below this, Jev isn't sure -> escalate
HIGH_RISK_FRACTION = 0.75 # risk score above this always escalates


RISK_CRITERIA = [
    "Trivial: docs, comments, formatting, renames.",
    "Low: minor logic changes confined to a single isolated function or module.",
    "Moderate: logic changes with a limited blast radius.",
    "High: auth, payments, data migrations, or public API contracts.",
    "Critical: core infrastructure, billing, or irreversible destructive operations.",
]
# `risk` is a Score, whose expected value ranges over [0, len(criteria) - 1],
# not [0, 1] like the Noul fields below -- threshold it accordingly.
HIGH_RISK_THRESHOLD = HIGH_RISK_FRACTION * (len(RISK_CRITERIA) - 1)


QUESTIONS = {
    "area": Choice(
        instructions="Which part of the codebase does this change mainly affect?",
        criteria={
            "frontend": "UI components, styles, client-side state.",
            "backend": "APIs, business logic, services, database queries.",
            "infra": "CI/CD, Docker, Terraform, deployment config.",
            "docs": "README, comments, documentation only.",
            "tests": "Test files only.",
        },
    ),
    "risk": Score(
        instructions="How risky is it to merge this change?",
        criteria=RISK_CRITERIA,
    ),
    "security": Noul(
        instructions="The change touches authentication, authorization, secrets, "
        "input validation, or other security-sensitive code.",
    ),
    "breaking": Noul(
        instructions="The change could break existing callers, clients, or stored data.",
    ),
    "tested": Noul(
        instructions="The diff adds or updates tests that cover the changed behavior.",
    ),
    "dependencies": Noul(
        instructions="The change adds, upgrades, downgrades, or removes a third-party "
        "dependency (package manifest, lockfile, or a new import of an external library).",
    ),
    "performance": Noul(
        instructions="The change could meaningfully affect runtime performance, memory "
        "usage, or database query cost (e.g. loops, N+1 queries, algorithmic complexity).",
    ),
    "secrets_exposure": Noul(
        instructions="The change touches environment variables, secrets, credentials, "
        "or configuration files in a way that could expose sensitive values.",
    ),
    "addresses_description": Noul(
        instructions="The `diff` actually implements what `description` claims the "
        "change does, with no unrelated scope creep and no missing pieces.",
    ),
}


def load_diff() -> str:
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            diff = f.read()
    else:
        diff = subprocess.run(
            ["git", "diff", f"{BASE_BRANCH}...HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout
    if not diff.strip():
        sys.exit("No changes to review.")
    return diff[:MAX_DIFF_CHARS]


def load_pr_description() -> str:
    """Stand-in for the PR body: commit messages unique to HEAD vs. BASE_BRANCH."""
    result = subprocess.run(
        ["git", "log", f"{BASE_BRANCH}..HEAD", "--format=%B"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def triage(diff: str, description: str):
    classifier = TypeSafeClassifier()
    questions = dict(QUESTIONS)
    state: dict[str, str] | str = diff
    if description:
        state = {"description": description, "diff": diff}
    else:
        questions.pop("addresses_description", None)
    start = time.perf_counter()
    result = classifier.invoke({"state": state, "questions": questions})
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def escalation_reasons(r) -> list[str]:
    reasons = []
    if r.nouls["security"].noul > FLAG_THRESHOLD:
        reasons.append("security-sensitive")
    if r.nouls["breaking"].noul > FLAG_THRESHOLD:
        reasons.append("possible breaking change")
    if r.nouls["tested"].noul < FLAG_THRESHOLD and r.choices["area"].choice != "docs":
        reasons.append("no test coverage")
    if r.scores["risk"].score > HIGH_RISK_THRESHOLD:
        reasons.append("high risk score")
    if r.nouls["dependencies"].noul > FLAG_THRESHOLD:
        reasons.append("dependency change")
    if r.nouls["performance"].noul > FLAG_THRESHOLD:
        reasons.append("possible performance impact")
    if r.nouls["secrets_exposure"].noul > FLAG_THRESHOLD:
        reasons.append("possible secrets/config exposure")
    addresses_description = r.nouls.get("addresses_description")
    if addresses_description is not None and addresses_description.noul < FLAG_THRESHOLD:
        reasons.append("diff may not match PR description")
    if r.choices["area"].confidence < MIN_CONFIDENCE:
        reasons.append("low triage confidence")
    if r.scores["risk"].confidence < MIN_CONFIDENCE:
        reasons.append("uncertain risk level")
    return reasons


def deep_review(diff: str, reasons: list[str]) -> str:
    client = anthropic.Anthropic()
    prompt = (
        "You are a senior engineer reviewing a pull request.\n"
        f"A fast triage model flagged it for: {', '.join(reasons)}.\n"
        "Focus on those concerns. Give concrete, line-referenced feedback "
        "and end with APPROVE or REQUEST CHANGES.\n\n"
        f"```diff\n{diff}\n```"
    )
    response = client.messages.create(
        model=REVIEW_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")


def main() -> None:
    diff = load_diff()
    description = load_pr_description()
    r, ms = triage(diff, description)

    area = r.choices["area"]
    print(f"\n⚡ Jev triage ({ms:.0f} ms)")
    if description:
        print(f"  description: {description.splitlines()[0][:80]}")
    else:
        print("  description: (none found — addresses_description check skipped)")
    print(f"  area      : {area.choice} (confidence {area.confidence:.2f})")
    print(f"  risk      : {r.scores['risk'].score:.2f}")
    print(f"  security  : {r.nouls['security'].noul:.2f}")
    print(f"  breaking  : {r.nouls['breaking'].noul:.2f}")
    print(f"  tested    : {r.nouls['tested'].noul:.2f}")
    print(f"  deps      : {r.nouls['dependencies'].noul:.2f}")
    print(f"  perf      : {r.nouls['performance'].noul:.2f}")
    print(f"  secrets   : {r.nouls['secrets_exposure'].noul:.2f}")
    addresses_description = r.nouls.get("addresses_description")
    if addresses_description is not None:
        print(f"  matches   : {addresses_description.noul:.2f}")

    reasons = escalation_reasons(r)
    if not reasons:
        print("\n✅ Low risk — fast-path approved, no LLM review needed.")
        return

    print(f"\n🔍 Escalating to {REVIEW_MODEL}: {', '.join(reasons)}\n")
    print(deep_review(diff, reasons))


if __name__ == "__main__":
    main()
