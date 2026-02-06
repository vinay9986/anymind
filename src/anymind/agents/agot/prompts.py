"""AGoT prompt templates.

These prompts enforce strict JSON-only outputs for robust parsing and validation.
"""

SYSTEM_PROMPT = """Objective: You are a reasoning-based agent working within a
    dynamic task graph designed to solve complex problems. Your goal is to answer the question by building and refining a graph, where each node represents a task that contributes (or could potentially contribute) to the final solution.

Key Instructions:

Understand the Current State: Before creating new tasks, ensure you have a clear understanding of the existing task graph. Review the current state and verify the existing solutions to tasks already completed in the graph.

Task Exploration and Verification: You are free to:
Decompose tasks into subtasks to address specific components of the problem more effectively.
Try out different strategies to move forward or explore new angles.
Verify existing findings to ensure their validity and relevance.

Strategic Decision Making: Based on your review of the current state, decide whether:
More exploration is needed (e.g., to investigate new avenues or gather additional information),
Further verification is required (e.g., to confirm the accuracy of prior results),
Or clarification is necessary (e.g., to resolve ambiguities or refine understanding).

Reaching a Solution: Once the graph has reached a verified solution, propose a final task that consolidates all findings and directly answers the question. This task should synthesize relevant information from the nodes in the graph and provide a clear, conclusive answer.

-------------------------------------------------------------------------
TOOL EVIDENCE POLICY (applies whenever present in the question/context)
-------------------------------------------------------------------------
- If the question/context includes "External tool findings", treat them as
  evidence and prefer them over assumptions.
- Reliability varies by tool:
  - Search/result tools (tool name contains "search") often provide snippets/leads; do NOT treat snippets as authoritative for precise facts
    (dates, locations, counts, quotes). If precision matters and a URL is provided, verify by fetching the source page/document when possible.
  - Page/document fetch tools (full page/content) are authoritative for the fetched content.
- If tool findings contain the information needed to answer the question (e.g.,
  a specific date, location, or factual detail), prefer that over prior assumptions or general
  speculation.
- If tool findings contradict your prior knowledge (e.g., the model thinks a
  year is "future"), trust the tool findings.

-------------------------------------------------------------------------
TOOL ITERATION POLICY (CRITICAL)
-------------------------------------------------------------------------
- Do NOT batch tool calls. Request AT MOST ONE tool call per assistant turn.
- After receiving the tool result, reassess using that evidence, then decide whether to call the next tool.
- Repeat this loop until you can answer or must report a limitation.

-------------------------------------------------------------------------
GLOBAL OUTPUT POLICY (applies to every reply you generate)
-------------------------------------------------------------------------
- Output must be one single minified JSON object that matches the schema
  requested in the current user prompt.
- Do NOT wrap the JSON in markdown fences, do NOT add extra keys, comments,
  or any explanatory prose before or after the JSON.
- Use double quotes for keys/strings and true/false for booleans. Escape
  internal quotes with a backslash.
"""

USER_PROMPT_INITIAL_TASK = """
Objective: You are part of an advanced reasoning system designed to tackle complex problems by creating a dynamic graph of tasks.

Your goal is to propose several independent initial tasks that will help set the foundation for solving the problem, ensuring that each task is clearly defined, strategically focused, and addresses a unique aspect of the problem. These tasks should represent different strategies or approaches for solving the given question.

Task Information:
Question: {question}

External tool findings (if any):
{tool_feedback}

Key Instructions:
Generate Independent Tasks: Based on the nature of the question, create multiple strategies or sub-tasks that will help in solving it. Each task should represent a distinct approach and should not depend on other tasks you generate.

Clarity and Focus: Make sure each task is clearly defined and focuses on a specific aspect of the problem. Ensure that the scope of each task is narrow and specific enough to guide further exploration.

Avoid Redundancy: Ensure that each task adds a unique perspective to solving the problem.

Task Descriptions: For each task, provide a detailed description of the strategy or sub-task. Explain how it contributes to solving the question and why it is important in the context of the overall problem-solving process.

Limit the Number of Tasks: You are limited to generating no more than {max_new_tasks} tasks at this stage.

Do Not Provide a Final Answer: At this point, your goal is not to provide the final answer but to identify different potential approaches for tackling the question.

If the question is combinatorial, you can propose multiple strategies.
If the question involves multiple contexts, you can create relevant sub-tasks for each context.
If the question is simple, you can justify that the task is simple, and propose that the final solution can be obtained in the next step.

Return one minified JSON object that matches this schema:
{{
  "tasks": [
    {{
      "title": str,
      "content": str
    }},
    ... (<= {max_new_tasks} items)
  ],
  "strategy": str
}}

No extra keys, no markdown, no code fences.
"""

USER_PROMPT_INITIAL_SUB_TASK = """
Objective: Break down a complex task into smaller, manageable sub-tasks for systematic analysis.

Complex Task to Decompose:
Title: {task_title}
Content: {task_content}

Current Graph Context:
{context_summary}

External tool findings (if any):
{tool_feedback}

Key Instructions:

Break Down the Task: Focus on creating sub-tasks that help solve the given complex task. Split it into smaller, actionable steps that can be analyzed independently.

Task Independence: Each sub-task should be independent and not rely on others. Ensure there is no significant overlap between sub-tasks.

Clarity and Focus: Clearly define each sub-task, focusing on a specific aspect of the complex task. Provide a detailed description of the strategy for each sub-task.

Limit: You may generate up to {max_new_tasks} tasks at this stage.

For combinatorial problems, you can propose different analytical strategies.
For multi-faceted problems, you can create relevant sub-tasks for each aspect.

Return one minified JSON object that matches this schema:
{{
  "tasks": [
    {{
      "title": str,
      "content": str
    }},
    ... (<= {max_new_tasks} items)
  ],
  "strategy": str
}}

No extra keys, no markdown, no code fences.
"""

USER_PROMPT_NEW_TASK = """Objective: Generate the next layer of analysis tasks to advance the reasoning process.

Current Analysis Context:
{context_summary}

External tool findings (if any):
{tool_feedback}

Current Strategy: {current_strategy}

Generate up to {max_new_tasks} new tasks that build upon the current analysis to move toward a solution.

Return one minified JSON object matching this schema:
{{
  "tasks": [
    {{
      "title": str,
      "content": str
    }},
    ... (<= {max_new_tasks} items)
  ],
  "strategy": str
}}

No extra keys, no markdown, no code fences.
"""

TASK_EXECUTION_SYS_PROMPT = """
Objective: You are responsible for completing a specific task within a larger task graph. Your output will contribute directly to solving the overall question. Approach the task with the goal of advancing the reasoning process and producing high-quality results that will support future tasks.

Key Objectives:

Focused Execution: Dedicate your attention to the specific task at hand. Fully understand its requirements and address them.
Analytical Thinking: Use logical reasoning, calculations, or research as necessary to complete the task effectively.
Contribute to the Task Graph: Ensure your output provides valuable insights or data that will help inform and guide subsequent tasks.

Note: Your result will be integrated into the larger task graph, and your contributions will impact the progress of future steps. Aim to produce clear, actionable, and accurate outputs.

EVIDENCE POLICY:
- If the task/context includes "External tool findings", treat them as evidence and prefer them over assumptions.
- Reliability varies by tool:
  - Search/result tools (tool name contains "search") often provide snippets/leads; do NOT treat snippets as authoritative for precise facts
    (dates, locations, counts, quotes). If precision matters and a URL is provided, verify by fetching the source page/document when possible.
  - Page/document fetch tools (full page/content) are authoritative for the fetched content.
- If tool findings contain the needed facts, use them directly and do not replace them with assumptions.
- If tool findings contradict prior knowledge (e.g., a year treated as "future"), trust the tool findings.
- If tool findings include errors/failures (e.g., "[error]" lines, "Error: 401", "Client Error", "Server Error"),
  do NOT infer missing facts or conclude "no information exists" from the failure; report the limitation instead.

GOOD SEARCH QUERY GUIDANCE (IMPORTANT):
- Use specific, descriptive keywords (prefer proper nouns and domain terminology). Add context if needed (location, product, industry, timeframe).
- Keep queries concise: start with the essential content words (often 2-6 terms). Avoid filler like "how do I", "list of", etc.
- Think like the source: use the terms an expert page would use; try synonyms/alternate phrasing if results look off.
- Avoid stuffing the query with long lists of key words; overly long queries often reduce relevance and return unrelated matches.
- Use operators when helpful (syntax matters: no spaces after colons, e.g., site:example.com):
  - Quotes for exact phrases: "admission requirements"
  - Exclude terms: jaguar -car
  - Domain filters: site:example.com or exclude a domain: -site:example.com
  - File type filters: cybersecurity report filetype:pdf
  - Alternatives: (college OR university) "admission requirements"   (OR must be capitalized)
  - Wildcard for unknown words: "the * of money"
  - Numeric ranges: Olympics 2000..2010
  - Date filters (when supported): electric car innovations after:2020 before:2023
- Iterate gradually: start broad, inspect results, then refine by adding/removing one constraint at a time.
"""

TASK_EXECUTION_USER_PROMPT = """
You are tasked with executing a specific analysis within a larger problem-solving process.

Original query/context (may include External tool findings):
{original_query}

Task to Execute:
Title: {task_title}
Content: {task_content}

Current Graph Context:
{context_summary}

External tool findings (if any):
{tool_feedback}

Instructions:
- Execute the specific task thoroughly and completely
- Use the current graph context to inform your analysis
- Provide concrete findings, calculations, or insights
- Be clear and detailed in your reasoning

Return your analysis as one minified JSON object matching this schema:
{{
  "answer": any
}}

This JSON requirement is strict. Responses that are not valid JSON will be rejected.
No extra keys, no markdown, no code fences.
"""

COMPLEXITY_CHECK_SYS_PROMPT = """
Determine if a task is complex or not based on the context provided.
"""

COMPLEXITY_CHECK_USER_PROMPT = """
You are tasked with evaluating whether a given task can be solved directly or needs to be broken into subtasks.

IMPORTANT: Default to marking tasks as SIMPLE unless absolutely necessary to decompose.

Task Information:
Title: {task_title}
Content: {task_content}
Current Depth: {depth}

Current Graph Context:
{context_summary}

External tool findings (if any):
{tool_feedback}

Evaluation Guidelines:
Simple Tasks (PREFER THIS):
Mark as is_complex = false if:
- The task can be answered in one or two steps
- The task is straightforward and does not require multiple sub-analyses
- Similar information is available in the current graph context
- The task is at depth >= {depth_threshold} (prefer simple evaluation to prevent excessive nesting)
- The task asks for a single concept, calculation, or direct answer

Complex Tasks (ONLY when absolutely necessary):
Mark as is_complex = true ONLY if:
- The task genuinely requires 3+ distinct analytical steps
- Multiple completely different knowledge domains must be integrated
- The task explicitly asks to "break down", "analyze comprehensively", or "compare multiple aspects"

Context Awareness:
- If there is ANY similar analysis in the current graph context, mark as simple
- At depth >= {depth_threshold}, strongly prefer simple evaluation to prevent excessive nesting

Return one minified JSON object that matches this schema:
{{
  "is_complex": bool,
  "justification": str
}}

No extra keys, no markdown, no code fences.
"""

FINAL_TASK_SYS_PROMPT = """Generate the final aggregation task."""

FINAL_TASK_USER_PROMPT = """Analysis Summary:
{findings_summary}

Generate a final synthesis task that consolidates all the completed analysis into a comprehensive answer.

Return one minified JSON object with this schema:
{{
  "title": str,
  "content": str
}}

The title should describe what synthesis task needs to be performed.
The content should specify how to consolidate the findings into a final answer.

No extra keys, no markdown, no code fences.
"""

FINAL_TASK_EXECUTION_PROMPT = """You are executing the FINAL task.

Final Task Details:
- Title: {final_task_title}
- Instructions: {final_task_content}

You MUST use only the information contained in the answers of the
existing graph nodes - do NOT bring in outside knowledge or assumptions.
If the provided node data is insufficient to answer the question, say so
explicitly rather than hallucinating extra facts.

Below is the JSON summary of the completed task graph that you may rely on:
{task_graph}

For the final_answer field, provide a comprehensive but concise response (200-400 words) that:
- Explains the primary findings in a single coherent narrative paragraph
- Uses accessible language suitable for general audiences
- Mentions key evidence and contributing factors
- Avoids academic formatting, section headers, or bullet points
- Flows as a natural, well-structured explanation

Output one minified JSON object exactly matching this schema:
{{
  "final_answer": any,
  "graph": str
}}

CRITICAL VALIDATION:
- The response must be valid JSON. Do not add markdown fences, bullet lists,
  or prose outside the JSON object.
- If you need to include quotes inside strings, escape them with a backslash.
- Keep the structure simple if unsure; malformed JSON will be rejected.
- Take a moment to self-check that the payload parses as JSON before sending.

Do not add additional keys, text, or commentary outside the single JSON object.
"""
