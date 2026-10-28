# src/llm_prompts.py

from __future__ import annotations


TERM_EXTRACTION_SYSTEM_PROMPT = """You are an expert terminology extractor for UAV / UAS / RPAS technical documents.

Your task is to extract domain-specific terms exactly as they appear in the text.

Important:
- Return ONLY extracted terms.
- Return exactly ONE term per line.
- Do NOT include explanations, commentary, headings, bullets, or numbering.
- Keep the original surface form from the text whenever possible.
- If both an acronym and its full form appear in the text, include BOTH on separate lines.
- Include technical phrases, acronyms, class labels, platform names, subsystem names, operational concepts, use-case phrases, and important formatted terms.
- Do NOT rewrite, summarize, normalize, or merge terms.
- Preserve capitalization, slashes, hyphens, underscores, and numeric ranges when they are part of the term.
"""


def build_term_extraction_user_prompt(text: str) -> str:
    return f"""Extract UAV-domain technical terms from the text below.

What counts as a valid term:
1. Acronyms and abbreviations:
   Examples: UAV, UAS, RPAS, GCS, ISR, HALE, MALE, VLA/BLOS, MA/A

2. Full expansions of acronyms:
   Examples: Unmanned aerial vehicle, Ground Control Station, High altitude long endurance

3. Platform / vehicle / product names:
   Examples: RQ-11B Raven, Shadow 200, Global Hawk, Puma AE, Camcopter S-100, Harop

4. Classes / category labels / type labels:
   Examples: Class I, Class II, Class III, Micro, Mini, Small, Large, Ultralight

5. Operational / regulatory / mission phrases:
   Examples: controlled airspace, operational altitude, restricted airspace, search and rescue,
   coast guard support, wildfire monitoring, traffic control tasks

6. Components / subsystems / sensors / infrastructure terms:
   Examples: Payload, Communication link, Launch and retrieval system, EO_Camera, IR_Camera

7. Structured technical expressions that appear exactly in the text:
   Examples: Long_Range_Acoustic_Device, 20–150 kg, <2 kg, Very low altitude (BLOS)

Extraction rules:
- Return one term per line.
- Return only terms from the text.
- Keep exact wording and formatting from the text.
- If both a short form and full form appear, include both.
- If singular and plural forms both appear in the text and both are meaningful terms, include both.
- Include multi-word terms, acronyms, named systems, and technical class labels.
- Exclude references, citations, author names, email addresses, URLs, page artifacts, table markup, and pure boilerplate.

Examples of desired output style:
UAV
Unmanned aerial vehicle
GCS
Ground Control Station
RPAS
Remotely piloted aircraft system
HALE
High altitude long endurance
Class I
search and rescue
RQ-11B Raven
Camcopter S-100
20–150 kg

Now extract the terms from this text.

TEXT:
{text}
"""