# SOP Agent Token Usage Report

**Log file:** `/var/folders/31/3n55rm_96yn9kz20gnk1jy780000gp/T/sop_agent-9a32ddbd6d664296adf487c306838c3b.log`
**Log size:** 51610554 bytes

## Executive Summary
- 107 LLM responses recorded; prompt tokens dominate total usage.
- Total tokens: **5,470,488** (prompt 5,455,169, completion 15,319).
- Prompt tokens are **99.72%** of total.
- Top 10 calls account for **2,300,502 tokens (42.05% of total)**.

## Token Usage Distribution (LLM Responses)
- Calls: **107**
- Prompt tokens — min/median/mean/max: **417 / 9703 / 50982 / 271943**
- Completion tokens — min/median/mean/max: **73 / 99 / 143 / 800**

## Prompt Size Indicators (LLM Requests)
- Requests logged: **108**
- Message count — min/median/mean/max: **2 / 3 / 22 / 94**
- Content size (chars) — min/median/mean/max: **1,556 / 38,440 / 200,015 / 1,084,792**
- Requests containing `Shared tool evidence`: **106** of **108**

## Top 10 Prompt-Heavy Calls

| Timestamp (UTC) | Model | Prompt Tokens | Completion Tokens |
|---|---|---:|---:|
| 2026-02-13T16:10:30.338287Z | gpt-5.1-2025-11-13 | 271,943 | 73 |
| 2026-02-13T16:10:13.886440Z | gpt-5.1-2025-11-13 | 262,116 | 73 |
| 2026-02-13T16:09:58.515345Z | gpt-5.1-2025-11-13 | 252,487 | 73 |
| 2026-02-13T16:09:50.124916Z | gpt-5.1-2025-11-13 | 243,049 | 73 |
| 2026-02-13T16:09:37.178190Z | gpt-5.1-2025-11-13 | 233,794 | 73 |
| 2026-02-13T16:09:28.140588Z | gpt-5.1-2025-11-13 | 224,731 | 73 |
| 2026-02-13T16:09:07.413149Z | gpt-5.1-2025-11-13 | 215,830 | 73 |
| 2026-02-13T16:08:48.879959Z | gpt-5.1-2025-11-13 | 207,093 | 73 |
| 2026-02-13T16:08:41.304742Z | gpt-5.1-2025-11-13 | 198,547 | 73 |
| 2026-02-13T16:08:34.955516Z | gpt-5.1-2025-11-13 | 190,182 | 73 |

## Key Observations
- The largest prompts exceed **200k prompt tokens**; output tokens remain tiny (~73).
- Requests include **large, repeated human messages** containing `Shared tool evidence` and the original query.
- Prompt sizes grow as more SOP nodes execute: message counts reach **94** and per-request content size exceeds **1.0M characters**.
- Token usage is dominated by a small number of very large prompts (top 20 calls = 69.12% of total).

## Likely Cause of Token Explosion
- SOP orchestration appears to append **full tool evidence** to the conversational history at each step.
- Each subsequent LLM call includes the entire accumulated history, so prompt tokens grow linearly (or worse) with step count.
- The evidence payload itself is large (tens of thousands of chars per step), and is repeated across many requests.

## What to Inspect Next (No Code Changes Yet)
- Confirm where SOP appends `Shared tool evidence` to each node input.
- Verify whether evidence is duplicated per node vs. referenced once.
- Check if tool outputs are being reinserted into history rather than stored externally and referenced by ID.