from __future__ import annotations

from typing import Any, Dict

from anymind.runtime.evidence import EvidenceLedger, EvidenceRecord
from anymind.runtime.messages import message_text
from anymind.runtime.usage import normalize_usage_metadata


async def render_with_citations(
    *,
    model_client: Any,
    model_name: str,
    draft: str,
    evidence_records: list[EvidenceRecord],
) -> tuple[str, Dict[str, Dict[str, int]]]:
    summary = EvidenceLedger.summarize(evidence_records)
    prompt = (
        "You are a response editor. Rewrite the draft to include citations in square brackets "
        "after claims that rely on evidence. Use only the evidence IDs provided. "
        "Do not invent citations. Keep the answer concise.\n\n"
        f"Draft:\n{draft}\n\nEvidence ledger:\n{summary}"
    )
    message = await model_client.ainvoke(
        [("system", "Add citations."), ("user", prompt)]
    )
    return message_text(message), normalize_usage_metadata(model_name, [message])
