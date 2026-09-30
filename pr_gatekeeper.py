"""
pr_gatekeeper.py — two-speed code review with Jev + an LLM.

System One (Jev): triages every pull request diff in one call — area, risk,
security impact, breaking changes, test coverage — and returns calibrated
probabilities instead of text.

System Two (LLM): only runs when Jev flags the PR as risky OR isn't confident.
Most PRs (docs, small fixes) never pay for a full LLM review.

Usage:
    pip install langchain-typesafe langchain-openai
    export TYPESAFE_API_KEY=...  OPENAI_API_KEY=...
    python pr_gatekeeper.py                 # diffs HEAD against origin/main
    python pr_gatekeeper.py path/to/pr.diff # or review a saved diff
"""

import os
import subprocess
import sys
import time

from langchain.chat_models import init_chat_model
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

REVIEW_MODEL = os.getenv("REVIEW_MODEL", "openai:gpt-5.6-terra")
BASE_BRANCH = os.getenv("BASE_BRANCH", "origin/main")
MAX_DIFF_CHARS = 20_000

FLAG_THRESHOLD = 0.5      # Noul probability that counts as "yes"
MIN_CONFIDENCE = 0.6      # below this, Jev isn't sure -> escalate
HIGH_RISK_THRESHOLD = 0.7 # risk score above this always escalates

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
        criteria=[
            "Trivial: docs, comments, formatting, renames.",
            "Moderate: logic changes with a limited blast radius.",
            "High: auth, payments, data migrations, or public API contracts.",
        ],
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


def triage(diff: str):
    classifier = TypeSafeClassifier()
    start = time.perf_counter()
    result = classifier.invoke({"state": diff, "questions": QUESTIONS})
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
    if r.choices["area"].confidence < MIN_CONFIDENCE:
        reasons.append("low triage confidence")
    if r.scores["risk"].confidence < MIN_CONFIDENCE:
        reasons.append("uncertain risk level")
    return reasons


def deep_review(diff: str, reasons: list[str]) -> str:
    llm = init_chat_model(REVIEW_MODEL)
    prompt = (
        "You are a senior engineer reviewing a pull request.\n"
        f"A fast triage model flagged it for: {', '.join(reasons)}.\n"
        "Focus on those concerns. Give concrete, line-referenced feedback "
        "and end with APPROVE or REQUEST CHANGES.\n\n"
        f"```diff\n{diff}\n```"
    )
    return llm.invoke(prompt).content


def main() -> None:
    diff = load_diff()
    r, ms = triage(diff)

    area = r.choices["area"]
    print(f"\n⚡ Jev triage ({ms:.0f} ms)")
    print(f"  area      : {area.choice} (confidence {area.confidence:.2f})")
    print(f"  risk      : {r.scores['risk'].score:.2f}")
    print(f"  security  : {r.nouls['security'].noul:.2f}")
    print(f"  breaking  : {r.nouls['breaking'].noul:.2f}")
    print(f"  tested    : {r.nouls['tested'].noul:.2f}")

    reasons = escalation_reasons(r)
    if not reasons:
        print("\n✅ Low risk — fast-path approved, no LLM review needed.")
        return

    print(f"\n🔍 Escalating to {REVIEW_MODEL}: {', '.join(reasons)}\n")
    print(deep_review(diff, reasons))


if __name__ == "__main__":
    main()
