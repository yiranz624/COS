"""Python-controlled retrieval state machine for evidence-heavy questions."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict
from typing import Any

from agent_harness.orchestration.context import OrchestrationContext
from orchestrations.contracts import CandidateV2, SubtaskPlanV1
from orchestrations.reference_v4 import (
    ConstraintV4,
    _constraint_instruction,
    _merge_candidates,
    _programmatic_admission,
)

MANIFEST = {
    "script_id": "reference_v5", "version": 5, "entrypoint": "solve",
    "input_schema": "string", "output_schema": "string",
    "compatible_runtime": ">=1.1",
    "roles": ("decomposer", "route_designer", "query_planner",
              "result_selector", "verifier", "synthesizer"),
}

_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_RESULT = re.compile(
    r"\[\d+\] Title: (?P<title>.*?)\n(?:    Date: .*?\n)?"
    r"(?:    Snippet: (?P<snippet>.*?)\n)?    URL: (?P<url>\S+)", re.S,
)


async def solve(ctx: OrchestrationContext) -> str:
    task = str(ctx.task)
    restored = ctx.load_checkpoint("reference_v5_state") or {}
    ctx.emit("script_started", **_identity(ctx))
    if restored.get("final_answer"):
        ctx.emit("script_completed", **_identity(ctx), restored=True)
        return str(restored["final_answer"])

    constraints = await _constraints(ctx, task, restored)
    state = {
        "constraints": [asdict(item) for item in constraints],
        "routes": list(restored.get("routes", [])),
        "executed_queries": list(restored.get("executed_queries", [])),
        "entities": list(restored.get("entities", [])),
        "candidates": list(restored.get("candidates", [])),
        "observations": list(restored.get("observations", [])),
        "verifications": list(restored.get("verifications", [])),
        "final_answer": None,
    }

    if not state["candidates"]:
        route_specs = state["routes"] or await _design_routes(ctx, task, constraints)
        state["routes"] = route_specs
        plans = await ctx.map(
            lambda route: _plan(
                ctx, task, _constraint_subset(constraints, route["active_constraints"]),
                route["route_id"], route["goal"], [], 2,
                expected_entity=route["pivot_entity_type"],
            ),
            route_specs,
        ) if route_specs else []
        observations = await _search_and_select(
            ctx, task, constraints, plans, state, round_index=0,
        )
        _merge_observations(state, observations)
        await _save(ctx, state)

    # Every AI-designed route owns its semantic pivot and expansion type.
    if not state["candidates"] and _can_spend(ctx, 12_000):
        expansions = []
        for route in state["routes"]:
            entities = [item for item in state["entities"]
                        if item.get("route_id") == route["route_id"]
                        and item.get("type") == route["pivot_entity_type"]
                        and item.get("matched_constraints")]
            if not entities:
                continue
            expansions.append(await _plan(
                ctx, task,
                _constraint_subset(constraints, route["deferred_constraints"]),
                f"{route['route_id']}:expansion",
                "Expand the supplied pivot entities toward concrete answer candidates "
                "while applying the deferred constraints.",
                state["executed_queries"], 3, entities=entities[:3],
                expected_entity=route["expansion_entity_type"],
            ))
        observations = await _search_and_select(
            ctx, task, constraints, expansions, state, round_index=1,
        )
        _merge_observations(state, observations)
        await _save(ctx, state)

    candidates = [CandidateV2.from_dict(item) for item in state["candidates"]]
    if not candidates:
        ctx.emit("verification_blocked", **_identity(ctx), reason="no_candidate")
        return await _finish(ctx, state, "信息未查全")

    if not state["verifications"]:
        verifications = await ctx.map(
            lambda candidate: _verify(ctx, task, constraints, candidate),
            candidates[:3],
        )
        state["verifications"] = verifications
        await _save(ctx, state)
    accepted = [item for item in state["verifications"] if item["accepted"]]
    if not accepted:
        ctx.emit("synthesis_blocked", **_identity(ctx),
                 reason="no_fully_verified_candidate")
        return await _finish(ctx, state, "信息未查全")

    ctx.emit("synthesis_started", **_identity(ctx),
             accepted_candidates=[item["answer"] for item in accepted])
    try:
        answer = str(await ctx.call(
            "synthesizer",
            f"原题：\n{task}\n\n程序化逐项验证通过的候选：\n"
            f"{json.dumps(accepted, ensure_ascii=False)}\n\n"
            "只输出benchmark要求的最短答案，不解释、不添加新事实。",
            key="reference_v5:synthesis", max_turns=1,
            metadata={"allowed_tools": []},
        )).strip() or "信息未查全"
    except Exception:
        answer = "信息未查全"
    ctx.emit("synthesis_completed", **_identity(ctx))
    return await _finish(ctx, state, answer)


async def _constraints(
    ctx: OrchestrationContext, task: str, restored: dict[str, Any],
) -> list[ConstraintV4]:
    if restored.get("constraints"):
        return [ConstraintV4(**item) for item in restored["constraints"]]
    ctx.emit("constraint_extraction_started", **_identity(ctx))
    try:
        raw = await ctx.call(
            "decomposer", _constraint_instruction(task),
            key="reference_v5:constraints", max_turns=1,
            metadata={"allowed_tools": []},
        )
        plan = SubtaskPlanV1.from_dict(raw)
        rows = [(criterion, "；".join(item.required_evidence) or "来源URL")
                for item in plan.subtasks for criterion in item.success_criteria]
        if not rows:
            raise ValueError("empty constraints")
        result = [ConstraintV4(f"C{i}", text, evidence)
                  for i, (text, evidence) in enumerate(rows[:8], 1)]
        fallback = False
    except (KeyError, TypeError, ValueError):
        result = [ConstraintV4("C1", "满足原题全部显式约束", "可核验URL")]
        fallback = True
    ctx.emit("constraint_extraction_completed", **_identity(ctx),
             constraint_count=len(result), fallback=fallback)
    return result


async def _design_routes(
    ctx: OrchestrationContext, task: str, constraints: list[ConstraintV4],
) -> list[dict[str, Any]]:
    try:
        raw = await ctx.call(
            "route_designer",
            f"原题：\n{task}\n\n约束："
            f"{json.dumps([asdict(item) for item in constraints], ensure_ascii=False)}\n\n"
            "自行判断最有辨识度的语义实体类型，生成至多3条互补路线。"
            "active_constraints只用于发现pivot；deferred_constraints留给扩展与验证。"
            "两组只能引用给出的C编号，不能加入题目不存在的假设。",
            key="reference_v5:route_design", max_turns=1,
            metadata={"allowed_tools": []},
        )
        values = raw.get("routes", []) if isinstance(raw, dict) else []
    except Exception:
        values = []
    routes = _validate_routes(values, constraints)
    ctx.emit("routes_designed", **_identity(ctx), routes=routes,
             route_count=len(routes))
    return routes


def _validate_routes(
    values: list[Any], constraints: list[ConstraintV4],
) -> list[dict[str, Any]]:
    known = {item.id for item in constraints}
    entity_types = {"person", "event", "work", "organization", "place",
                    "record", "answer"}
    routes, seen = [], set()
    for raw in values[:3]:
        if not isinstance(raw, dict):
            continue
        route_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(raw.get("route_id", "")))[:50]
        active = [str(x).upper() for x in raw.get("active_constraints", [])
                  if str(x).upper() in known]
        deferred = [str(x).upper() for x in raw.get("deferred_constraints", [])
                    if str(x).upper() in known and str(x).upper() not in active]
        pivot, expansion = (str(raw.get("pivot_entity_type", "")),
                            str(raw.get("expansion_entity_type", "")))
        if (not route_id or route_id in seen or not active
                or pivot not in entity_types or expansion not in entity_types):
            continue
        # Every required constraint remains visible in one of the two stages.
        deferred.extend(sorted(known - set(active) - set(deferred)))
        seen.add(route_id)
        routes.append({"route_id": route_id, "goal": str(raw.get("goal", "")),
                       "pivot_entity_type": pivot,
                       "expansion_entity_type": expansion,
                       "active_constraints": active,
                       "deferred_constraints": deferred})
    return routes


def _constraint_subset(
    constraints: list[ConstraintV4], ids: list[str],
) -> list[ConstraintV4]:
    selected = set(ids)
    return [item for item in constraints if item.id in selected]


async def _plan(
    ctx: OrchestrationContext, task: str, constraints: list[ConstraintV4],
    route_id: str, goal: str, prior_queries: list[str], limit: int, *,
    entities: list[dict[str, Any]] | None = None,
    expected_entity: str,
) -> dict[str, Any]:
    prompt = (
        f"原题：\n{task}\n\n约束：{json.dumps([asdict(x) for x in constraints], ensure_ascii=False)}"
        f"\n\n路线：{route_id}\n目标：{goal}\n已执行查询："
        f"{json.dumps(prior_queries, ensure_ascii=False)}\n已发现实体："
        f"{json.dumps(entities or [], ensure_ascii=False)}\n"
        f"生成至多{limit}条彼此不同的英文查询。每条expected_entity必须严格为"
        f"{expected_entity!r}。query必须直接服务当前目标；不要回答原题。"
    )
    try:
        raw = await ctx.call(
            "query_planner", prompt, key=f"reference_v5:plan:{route_id}",
            max_turns=1, metadata={"allowed_tools": []},
        )
        raw_queries = raw.get("queries", []) if isinstance(raw, dict) else []
    except Exception:
        raw_queries = []
    queries = _validate_queries(
        raw_queries, task, prior_queries, limit, expected_entity,
    )
    if not queries:
        ctx.emit("query_plan_rejected", **_identity(ctx), route_id=route_id,
                 reason="planner_returned_no_valid_query")
    ctx.emit("query_plan_created", **_identity(ctx), route_id=route_id,
             queries=queries)
    return {"route_id": route_id, "queries": queries,
            "expected_entity": expected_entity}


def _validate_queries(
    values: list[Any], task: str, prior: list[str], limit: int,
    expected_entity: str,
) -> list[dict[str, str]]:
    seen = {" ".join(item.casefold().split()) for item in prior}
    allowed_years = _allowed_years(task)
    result = []
    for value in values:
        if not isinstance(value, dict):
            continue
        query = " ".join(str(value.get("query", "")).split())
        normalized = query.casefold()
        if not 8 <= len(query) <= 240 or normalized in seen:
            continue
        if set(_YEAR.findall(query)) - allowed_years:
            continue
        if str(value.get("expected_entity", "")) != expected_entity:
            continue
        seen.add(normalized)
        result.append({"query": query, "goal": str(value.get("goal", "")),
                       "expected_entity": str(value.get("expected_entity", "record"))})
        if len(result) >= limit:
            break
    return result


async def _search_and_select(
    ctx: OrchestrationContext, task: str, constraints: list[ConstraintV4],
    routes: list[dict[str, Any]], state: dict[str, Any], *, round_index: int,
) -> list[dict[str, Any]]:
    prior = {" ".join(item.casefold().split()) for item in state["executed_queries"]}
    work = []
    for route in routes:
        for item in route["queries"]:
            normalized = " ".join(item["query"].casefold().split())
            if normalized not in prior:
                prior.add(normalized)
                state["executed_queries"].append(item["query"])
                work.append((route["route_id"], item))
    async def search(item: tuple[str, dict[str, str]]) -> dict[str, Any]:
        route_id, query = item
        try:
            raw = await ctx.tool(
                "web_search", {"q": query["query"], "num": 5}, role="solver",
                key=f"reference_v5:search:{round_index}:{_digest(query['query'])}",
            )
            records = _parse_search(str(raw))
            error = None
        except Exception as exc:
            records, error = [], f"{type(exc).__name__}: {exc}"
        ctx.emit("search_completed", **_identity(ctx), route_id=route_id,
                 query=query["query"], result_count=len(records), error=error)
        return {"route_id": route_id, "query": query, "results": records}
    batches = await ctx.map(search, work) if work else []
    grouped: dict[str, list[dict[str, Any]]] = {}
    expected_by_route = {route["route_id"]: route["expected_entity"]
                         for route in routes}
    for batch in batches:
        grouped.setdefault(batch["route_id"], []).append(batch)
    observations = []
    for route_id, payload in grouped.items():
        try:
            observation = await ctx.call(
                "result_selector", _selector_prompt(
                    task, constraints, payload, expected_by_route[route_id],
                ),
                key=f"reference_v5:select:{round_index}:{route_id}", max_turns=1,
                metadata={"allowed_tools": []},
            )
            if not isinstance(observation, dict):
                raise TypeError("selector output is not an object")
        except Exception as exc:
            observation = {"entities": [], "candidates": [],
                           "information_gain": False,
                           "missing_constraints": [f"选择器失败: {exc}"]}
        observation = _sanitize_observation(
            observation, route_id, expected_by_route[route_id], constraints,
        )
        observations.append(observation)
        ctx.emit("observation_created", **_identity(ctx), route_id=route_id,
                 entity_count=len(observation.get("entities", [])),
                 candidate_count=len(observation.get("candidates", [])),
                 information_gain=bool(observation.get("information_gain")))
    return observations


def _parse_search(value: str) -> list[dict[str, str]]:
    return [{key: (match.group(key) or "").strip()
             for key in ("title", "snippet", "url")}
            for match in _RESULT.finditer(value)][:5]


def _allowed_years(task: str) -> set[str]:
    years = [int(item) for item in _YEAR.findall(task)]
    allowed = {str(item) for item in years}
    lowered = task.casefold()
    if (len(years) >= 2 and any(marker in lowered for marker in
            ("between", "inclusive", "from", "range", "至", "到"))):
        low, high = min(years), max(years)
        if 0 <= high - low <= 100:
            allowed.update(str(item) for item in range(low, high + 1))
    return allowed


def _selector_prompt(
    task: str, constraints: list[ConstraintV4], payload: list[dict[str, Any]],
    expected_entity: str,
) -> str:
    return (
        f"原题：\n{task}\n\n约束：{json.dumps([asdict(x) for x in constraints], ensure_ascii=False)}"
        f"\n\n本轮检索结果：\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        f"本路线只允许抽取type={expected_entity!r}的中间实体。"
        "只从结果中抽取有URL支持的具体中间实体或答案候选。中间实体例如人物、作品、"
        "赛事或记录；candidate.answer必须是原题要求的答案形式，不能把裁判等中间实体"
        "当答案。每个实体的matched_constraints必须填写搜索摘要直接支持的C编号；"
        "不能证明裁判国籍时，不得填写C2。constraint_status使用C编号。"
    )


def _merge_observations(state: dict[str, Any], observations: list[dict[str, Any]]) -> None:
    state["observations"].extend(observations)
    entities = {(" ".join(str(x.get("name", "")).casefold().split()),
                 str(x.get("type", ""))): x for x in state["entities"]}
    candidates = [CandidateV2.from_dict(x) for x in state["candidates"]]
    for observation in observations:
        for entity in observation.get("entities", []):
            if isinstance(entity, dict) and entity.get("name"):
                entities[(" ".join(str(entity["name"]).casefold().split()),
                          str(entity.get("type", "")))] = entity
        candidates.extend(CandidateV2.from_dict(x)
                          for x in observation.get("candidates", [])
                          if isinstance(x, dict) and x.get("answer"))
    state["entities"] = list(entities.values())[:8]
    state["candidates"] = [asdict(x) for x in _merge_candidates(candidates)]


def _sanitize_observation(
    observation: dict[str, Any], route_id: str, expected_entity: str,
    constraints: list[ConstraintV4],
) -> dict[str, Any]:
    known = {item.id for item in constraints}
    entities = []
    for raw in observation.get("entities", []):
        if not isinstance(raw, dict) or raw.get("type") != expected_entity:
            continue
        matched = [str(item).upper() for item in raw.get("matched_constraints", [])
                   if str(item).upper() in known]
        if not raw.get("name") or not str(raw.get("url", "")).startswith("http"):
            continue
        if not matched:
            continue
        entities.append({"type": expected_entity, "name": str(raw["name"]),
                         "url": str(raw["url"]),
                         "matched_constraints": matched,
                         "route_id": route_id})
    candidates = [item for item in observation.get("candidates", [])
                  if isinstance(item, dict) and item.get("answer")]
    result = dict(observation)
    result["entities"] = entities
    result["candidates"] = candidates
    result["information_gain"] = bool(entities or candidates)
    result["route_id"] = route_id
    return result




async def _verify(
    ctx: OrchestrationContext, task: str, constraints: list[ConstraintV4],
    candidate: CandidateV2,
) -> dict[str, Any]:
    urls = [item.url for item in candidate.evidence if item.url.startswith("http")][:2]
    fetched: list[dict[str, str]] = []
    for index, url in enumerate(urls):
        try:
            content = await ctx.tool(
                "web_fetch", {"url": url, "info_to_extract":
                    f"Verify candidate {candidate.answer} against: " +
                    "; ".join(f"{x.id} {x.description}" for x in constraints)},
                role="verifier",
                key=f"reference_v5:fetch:{_digest(candidate.answer)}:{index}",
            )
            fetched.append({"url": url, "content": str(content), "error": ""})
        except Exception as exc:
            fetched.append({"url": url, "content": "", "error": str(exc)})
    try:
        raw = await ctx.call(
            "verifier", f"原题：\n{task}\n\n候选：{json.dumps(asdict(candidate), ensure_ascii=False)}"
            f"\n\n约束：{json.dumps([asdict(x) for x in constraints], ensure_ascii=False)}"
            f"\n\n已抓取证据：{json.dumps(fetched, ensure_ascii=False)}\n\n"
            "不得调用工具。逐项把已支持约束写成'C1: supported'；明确矛盾写入"
            "failed_criteria；证据不足的C编号两边都不写。accepted字段不会被采信。",
            key=f"reference_v5:verify:{_digest(candidate.answer)}", max_turns=1,
            metadata={"allowed_tools": []},
        )
        verdict = raw if isinstance(raw, dict) else {}
    except Exception as exc:
        verdict = {"satisfied_criteria": [], "failed_criteria": [],
                   "verified_evidence": [], "critique": str(exc), "confidence": 0}
    accepted, statuses, reasons = _programmatic_admission(
        constraints, candidate, verdict,
    )
    successful_urls = {item["url"] for item in fetched if not item["error"]}
    verifier_urls = {
        str(item.get("url", "")) for item in verdict.get("verified_evidence", [])
        if isinstance(item, dict) and item.get("status") == "verified"
    }
    if not successful_urls.intersection(verifier_urls):
        accepted = False
        reasons = [*reasons, "缺少成功Fetch且由Verifier标记verified的候选URL"]
    result = {"answer": candidate.answer, "accepted": accepted,
              "criterion_status": statuses, "reasons": reasons,
              "verdict": verdict, "fetched_evidence": fetched}
    ctx.emit("candidate_verification_completed", **_identity(ctx),
             answer=candidate.answer, accepted=accepted,
             criterion_status=statuses, rejection_reasons=reasons)
    return result


async def _save(ctx: OrchestrationContext, state: dict[str, Any]) -> None:
    await ctx.checkpoint("reference_v5_state", state)


async def _finish(
    ctx: OrchestrationContext, state: dict[str, Any], answer: str,
) -> str:
    state["final_answer"] = answer
    await _save(ctx, state)
    ctx.emit("script_completed", **_identity(ctx), restored=False)
    return answer


def _can_spend(ctx: OrchestrationContext, estimate: int) -> bool:
    maximum = ctx.metadata.get("budget", {}).get("max_tokens")
    return maximum is None or (
        ctx.remaining_budget().usage["tokens"] + estimate <= int(maximum * 0.85)
    )


def _identity(ctx: OrchestrationContext) -> dict[str, Any]:
    return {"script_id": ctx.metadata.get("script_id", "reference_v5"),
            "version": ctx.metadata.get("script_version", 5),
            "source_hash": ctx.metadata.get("script_source_hash", "")}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]
