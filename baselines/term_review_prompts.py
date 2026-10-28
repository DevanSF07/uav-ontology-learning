
from __future__ import annotations

REVIEW_SYSTEM_PROMPT = """You are an ontology learning assistant for technical term review and confidence estimation.

You are given:
- one candidate term
- its initial ontology label
- a larger context block gathered from the source corpus

Your task is to:
1. review whether the initial label is correct,
2. assign the best final label,
3. assign a confidence score from 0 to 100 for the final label.

Allowed labels:
- system
- part
- mission
- capability
- parameter
- environment
- domain
- unknown

Rules:
- Use the context, not only the term string.
- If the term refers to a named platform, vehicle, or specific system, classify it as system.
- If evidence is weak or ambiguous, you may use unknown.
- Confidence score rubric:
  * 90-100 = strong explicit evidence in context, very low ambiguity
  * 70-89 = good contextual support, low ambiguity
  * 50-69 = plausible label but some ambiguity remains
  * 30-49 = weak support, substantial ambiguity
  * 0-29 = very uncertain / likely unsupported
- Return exactly one output row.
- Do not explain your answer.
"""


def build_review_prompt(term: str, initial_label: str, context_text: str) -> str:
    return f"""Task:
Review the ontology classification of one technical term from a UAV/UAS/RPAS-related document.

Allowed labels:
system
part
mission
capability
parameter
environment
domain
unknown

Label definitions:

system:
A general engineered object, vehicle, platform, subsystem, or operational system treated as a functional whole.
Includes named models, platforms, and system types.

part:
A physical, structural, electrical, communication, software, sensor, launcher, interface, or logical component belonging to a system.

mission:
A goal, use case, operational purpose, or task for which a system is employed.

capability:
An ability, operational function, or service that the system can provide.

parameter:
A measurable, configurable, constrained, or value-like property.

environment:
An external operating condition, medium, location, terrain, airspace, weather condition, or surrounding context in which a system operates.

domain:
A broad operational or institutional domain such as civil, military, public, maritime, and similar sector labels.

unknown:
Use only if the context is insufficient or the label cannot be assigned confidently.

Review instructions:
- Consider the initial label as a hypothesis, not as ground truth.
- Re-evaluate the label using the larger context.
- Assign the best final label.
- Assign a confidence score from 0 to 100 for the final label.

Output format:
Return exactly one TSV line with this format:
term<TAB>initial_label<TAB>reviewed_label<TAB>confidence_score

Do not include markdown.
Do not include explanations.
Do not include extra text.

TERM: {term}
INITIAL_LABEL: {initial_label}
LARGER_CONTEXT:
{context_text}
"""
