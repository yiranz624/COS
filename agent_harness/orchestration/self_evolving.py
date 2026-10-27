"""State contracts for bounded self-evolving generated orchestration."""

from __future__ import annotations

from typing import Any


MAX_EVOLUTION_ROUNDS = 5


def normalize_failure_report(
    value: Any,
    *,
    failure_stage: str,
    summary: str,
    previous_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    previous_routes = (
        previous_report.get("failed_routes", [])
        if isinstance(previous_report, dict) else []
    )
    current_routes = raw.get("failed_routes", [])
    routes: list[str] = []
    for item in [*previous_routes, *(
        current_routes if isinstance(current_routes, list) else []
    )]:
        text = str(item).strip()
        if text and text not in routes and len(routes) < 20:
            routes.append(text)

    def list_value(name: str) -> list[Any]:
        item = raw.get(name, [])
        return list(item[:20]) if isinstance(item, list) else []

    return {
        "failure_stage": str(raw.get("failure_stage") or failure_stage),
        "summary": str(raw.get("summary") or summary)[:4000],
        "failed_routes": routes,
        "rejected_candidates": list_value("rejected_candidates"),
        "inheritable_facts": list_value("inheritable_facts"),
        "disputed_facts": list_value("disputed_facts"),
        "unverified_identity_conditions": list_value(
            "unverified_identity_conditions",
        ),
        "evidence_conflicts": list_value("evidence_conflicts"),
        "verifier_critique": str(raw.get("verifier_critique", ""))[:8000],
        "next_round_priorities": list_value("next_round_priorities"),
        "continue_evolution": bool(raw.get("continue_evolution", True)),
    }


def merge_fact_memory(
    current: list[dict[str, Any]],
    additions: list[Any],
    *,
    round_number: int,
    failure_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Merge source-backed facts and retain contradictions as disputed history."""
    merged = [dict(item) for item in current if isinstance(item, dict)]
    by_claim = {
        _claim_key(item.get("claim")): item
        for item in merged if _claim_key(item.get("claim"))
    }
    for raw in additions[:32]:
        if not isinstance(raw, dict):
            continue
        claim = str(raw.get("claim", "")).strip()
        urls = raw.get("source_urls", [])
        if not isinstance(urls, list):
            urls = []
        urls = [str(url).strip() for url in urls[:8] if str(url).strip()]
        if not claim or not urls:
            continue
        candidate_ids = raw.get("candidate_ids", [])
        if not isinstance(candidate_ids, list):
            candidate_ids = []
        key = _claim_key(claim)
        existing = by_claim.get(key)
        if existing is None:
            existing = {
                "claim": claim,
                "source_urls": urls,
                "candidate_ids": [str(item) for item in candidate_ids[:8]],
                "origin_round": round_number,
                "status": "active",
            }
            merged.append(existing)
            by_claim[key] = existing
        else:
            for url in urls:
                if url not in existing.setdefault("source_urls", []):
                    existing["source_urls"].append(url)
            for candidate_id in candidate_ids[:8]:
                cid = str(candidate_id)
                if cid not in existing.setdefault("candidate_ids", []):
                    existing["candidate_ids"].append(cid)
    if failure_report:
        disputed_claims = {
            _claim_key(
                item.get("claim", "") if isinstance(item, dict) else item,
            )
            for item in failure_report.get("disputed_facts", [])
        }
        rejected_ids = {
            str(item.get("candidate_id", "") if isinstance(item, dict) else item)
            for item in failure_report.get("rejected_candidates", [])
        }
        for fact in merged:
            fact_ids = {str(item) for item in fact.get("candidate_ids", [])}
            if _claim_key(fact.get("claim")) in disputed_claims or (
                fact_ids and fact_ids.intersection(rejected_ids)
            ):
                fact["status"] = "disputed"
    return merged[:64]


def evolution_context(
    facts: list[dict[str, Any]],
    previous_failure_report: dict[str, Any] | None,
    *,
    round_number: int,
) -> dict[str, Any]:
    """Return exactly the cross-round state allowed into a new generation."""
    return {
        "round": round_number,
        "active_facts": [
            dict(item) for item in facts
            if item.get("status") == "active"
        ],
        "previous_failure_report": previous_failure_report,
        "prohibited_inputs": [
            "previous Python source",
            "previous natural-language plan",
            "previous raw trace",
        ],
    }


def _claim_key(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


__all__ = [
    "MAX_EVOLUTION_ROUNDS", "evolution_context", "merge_fact_memory",
    "normalize_failure_report",
]
