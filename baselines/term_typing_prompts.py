from __future__ import annotations

SYSTEM_PROMPT = """You are an ontology learning assistant for technical term typing.

Your task is to assign exactly one ontology label to each candidate term, using both the term and its context.

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
- Use the context passage to determine the best label.
- Do not classify from the term string alone if the context changes the meaning.
- If the term is a named model, platform, vehicle, branded system, prototype, or specific system, classify it as system.
- Do not invent any extra labels.
- Return exactly one label per term.
- If the context is insufficient or the term is noisy/vague, return unknown.
- Do not explain the answers.
"""


def build_typing_prompt(term_context_rows: list[dict]) -> str:
    """
    term_context_rows: list of dicts with keys:
      - id
      - term
      - context

    Output format is TSV to avoid comma ambiguity inside terms/context.
    The model must return:
      id<TAB>term<TAB>label
    one row per input term.
    """
    header = """Task:
Given candidate technical terms extracted from a UAV/UAS/RPAS-related technical document, classify each term into exactly one label.

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
A general engineered object, vehicle, platform, device, subsystem, or operational system treated as a functional whole.
Use system for:
- system types
- categories
- variants
- subtypes
- named platforms
- named vehicles
- named system models

part:
A physical, structural, electrical, communication, software, sensor, launcher, interface, or logical component that belongs to or is contained in a system.

mission:
A goal, use case, operational purpose, or task for which a system is employed.
Ask: why is the system being used?

capability:
An ability, operational function, or service the system can provide.
Ask: what can the system do?

parameter:
A measurable, configurable, constrained, or value-like property.
Ask: how much / how fast / how large / what value / what range?

environment:
An external operating condition, medium, location, terrain, airspace, weather condition, or surrounding context in which a system operates.

domain:
A broad operational or institutional domain such as civil, military, public, maritime, etc.

unknown:
Use only if the context is insufficient or the term does not clearly fit any allowed label.

Methodology:
1. Use BOTH the term and the context.
2. Prefer the label best supported by context.
3. Terms may already be normalized or lemmatized.
4. If a term is instance-like but clearly refers to a named system/platform, label it as system.

Return format:
Return ONLY TSV lines in exactly this format:
id<TAB>term<TAB>label

Do not include markdown.
Do not include explanations.
Do not include extra text.
"""

    body_lines = [header, "Items to classify:"]
    for row in term_context_rows:
        body_lines.append(f"ID: {row['id']}")
        body_lines.append(f"TERM: {row['term']}")
        body_lines.append(f"CONTEXT: {row['context']}")
        body_lines.append("")

    return "\n".join(body_lines)