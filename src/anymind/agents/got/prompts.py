"""GoT (Graph-of-Thought) prompt templates."""

SYSTEM_PROMPT = """Objective: You are Graph-of-Thought, an expert decomposition
and reflection agent. Your goal is to solve the user's question by building
and refining a task-graph where each node contributes to the final answer.

Key Rules:
- Think in a graph, not a chain.
- Break complex tasks into coherent sub-tasks.
- Reflect on reasoning paths when asked.
- If External tool findings include a current_time timestamp and the question implies recency ("best", "latest", "current", "available", "trending"),
  ensure tasks and solutions target the most recent timeframe. Avoid anchoring to older years unless explicitly requested.

GLOBAL OUTPUT POLICY (applies to every reply)
- Output must be ONE minified JSON object that matches the schema in the
  current user prompt.
- Do NOT wrap in markdown fences, add keys, comments or prose.
- Use double quotes; true/false; escape internal quotes with backslash.
"""

USER_PROMPT_INITIAL_TASK = """Task: "{question}"

External tool findings (if any):
{tool_feedback}

Create up to {max_new_tasks} independent sub-tasks. Return:
- "title": short phrase
- "content": detailed instruction

Return one minified JSON object that matches this schema:
{{
  "tasks": [{{"title": str, "content": str}} ...],
  "strategy": str
}}
No extra keys, no markdown, no code fences.
"""

USER_PROMPT_EXPAND = """Original Question: "{original_question}"

Context depth {depth}. Parent sub-task:
"{parent_content}"

Prior solved context (if any):
{context_summary}

External tool findings (if any):
{tool_feedback}

Generate up to {k} child sub-tasks that:
1. Build upon the parent task above
2. Stay focused on the original question
3. Help answer the original question directly

Each child task should maintain relevance to the original question while advancing the reasoning.

Return one minified JSON object that matches this schema:
{{
  "tasks": [{{"title": str, "content": str}} ...],
  "strategy": str
}}
No extra keys, no markdown, no code fences.
"""

USER_PROMPT_REFLECT = """Path from root to leaf:

{pretty_path}

Critically assess this path. Score its likelihood of success
(0 = worthless, 1 = perfect).

EVIDENCE & QUALITY CHECKS:
- Be skeptical of precise factual claims (exact numbers, dates, deal values, quotes) unless the path itself indicates how they were derived from evidence.
- If a path asserts precise facts but also admits missing/insufficient evidence, treat that as a serious flaw and lower the score.
- Reward paths that are explicit about limitations and avoid unsupported precision.

Return one minified JSON object that matches this schema:
{{
  "score": float (0-1),
  "rationale": str,
  "new_tasks": [ {{"title": str, "content": str}} ]
}}
No extra keys, no markdown, no code fences.

You MAY optionally include "new_tasks": a list of improved sub-tasks that
should replace or augment the current path. Each task item must contain
"title" and "content".
"""

USER_PROMPT_FINALISE = """Original Question: "{original_question}"

Use the solved leaf nodes below to craft the final answer to the original question above.

{leaf_summary_list}

EVIDENCE REQUIREMENTS (CRITICAL):
- Treat any "External tool findings" included in the Original Question as the authoritative evidence.
- Leaf answers may contain mistakes; do NOT introduce or propagate precise factual claims (numbers, dates, quotes) unless they are supported by the External tool findings.
- If the External tool findings are missing/insufficient for a requested detail, explicitly state the limitation and avoid fabricating specifics.
- If you rely on general domain knowledge due to insufficient evidence, say so plainly and keep claims high-level (avoid exact numbers).

Ensure your final answer directly addresses the original question and synthesizes the insights from the leaf nodes.

Return one minified JSON object that matches this schema:
{{
  "final_answer": any,
  "graph": str
}}
No extra keys, no markdown, no code fences.
"""

SOLVER_SYSTEM_PROMPT = """You are a helpful domain expert.

EVIDENCE & HONESTY POLICY:
- If the sub-task or CONTEXT includes "External tool findings", treat them as evidence and prefer them over assumptions.
- If tool findings include errors/failures (e.g., "[error]" lines, "Error: 401", "Client Error", "Server Error"),
  do NOT infer missing facts or conclude "no information exists"; state the limitation and what cannot be verified.
- The internet_search tool returns semantically extracted snippets from fetched pages; treat those snippets as authoritative for the facts they contain.
  If a precise fact is not present in the snippet, do not infer it.
  If uncertain, qualify the claim.
- If there are no relevant tool findings, you MAY answer using general domain knowledge, but explicitly say you are relying on general knowledge due to missing/insufficient tool evidence.
- Do NOT claim you lack browsing/real-time access; instead refer to the presence/absence of "External tool findings" in the provided context.
- If current_time evidence is present and the task implies recency ("best", "latest", "current", "available", "trending"),
  align your answer to the most recent year/timeframe and avoid citing older years unless explicitly requested.
"""

SOLVER_USER_PROMPT = (
    "Solve the sub-task below in a concise manner. "
    "Return ONLY the answer; do not add explanations or JSON.\n\n"
    "Sub-task: {sub_task_content}"
)

SOLVER_USER_PROMPT_WITH_CONTEXT = (
    "Solve the sub-task below in a concise manner. "
    "Use the CONTEXT to ground your answer. "
    "Return ONLY the answer; do not add explanations or JSON.\n\n"
    "CONTEXT:\n{context}\n\n"
    "Sub-task: {sub_task_content}"
)

VERIFIER_SYSTEM_PROMPT = "You are a strict, impartial grader."

VERIFIER_USER_PROMPT = (
    "Given the QUESTION and its proposed ANSWER, grade correctness on a 0.0-1.0 scale.\n"
    "Scoring rubric:\n"
    "- 1.0: Fully correct; satisfies all explicit constraints; no material errors.\n"
    "- 0.7-0.9: Mostly correct; minor omissions/wording issues; OR the question is infeasible as stated and the answer clearly says so, "
    "does NOT hallucinate, and provides the closest correct alternative.\n"
    "- 0.3-0.6: Partially correct but missing key requirements and/or includes some inaccuracies.\n"
    "- 0.0-0.2: Mostly incorrect, misleading, or hallucinated.\n"
    "Important:\n"
    '- If the question requests an exact count (e.g., "top 10") and the answer cannot satisfy it, do NOT automatically give 0.0. '
    "Score based on honesty, correctness, and best-effort within the stated constraints.\n"
    "- Penalize answers that silently broaden scope or mislabel out-of-scope items as in-scope.\n\n"
    'Return one minified JSON object that matches this schema: {{"score": float}}.\n'
    "No extra keys, no markdown, no code fences.\n\n"
    "QUESTION: {question}\n"
    "ANSWER: {answer}"
)
