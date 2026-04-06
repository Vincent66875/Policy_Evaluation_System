import argparse
import asyncio
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=API_KEY)

DEFAULT_TRIALS = 30
DEFAULT_SEED = 7

# You can tune these if your empirical results suggest different behavior.
PERSONA_BEHAVIOR = {
    "Human_Avg": {
        "temperature": 0.65,
        "jargon_penalty": 14,
        "skimming_bias": 0.55,
        "omission_bias": 0.18,
        "overselect_bias": 0.08,
        "uncertainty_bias": 0.35,
        "d_threshold": 54,
        "base_threshold": 58,
        "noise": 9.5,
    },
    "Expert": {
        "temperature": 0.20,
        "jargon_penalty": 0,
        "skimming_bias": 0.10,
        "omission_bias": 0.04,
        "overselect_bias": 0.02,
        "uncertainty_bias": 0.10,
        "d_threshold": 30,
        "base_threshold": 42,
        "noise": 3.0,
    },
    "Non_Expert_Seniors": {
        "temperature": 0.55,
        "jargon_penalty": 20,
        "skimming_bias": 0.25,
        "omission_bias": 0.24,
        "overselect_bias": 0.14,
        "uncertainty_bias": 0.55,
        "d_threshold": 60,
        "base_threshold": 62,
        "noise": 8.0,
    },
}

PERSONAS = {
    "Human_Avg": {
        "profile": "45-year-old user with no IT expertise; skims policies.",
        "reading_style": "Skims text, notices obvious wording, often misses technical terms.",
        "belief_seed": (
            "I may not understand technical standards. If a term looks like a code or certification name, "
            "I may not know exactly what it means."
        ),
    },
    "Expert": {
        "profile": "IT professional with 10+ years of experience.",
        "reading_style": "Reads carefully and recognizes security terminology.",
        "belief_seed": (
            "I understand certification and security terminology. I look for explicit support in the policy text."
        ),
    },
    "Non_Expert_Seniors": {
        "profile": "70-year-old with secondary education; careful reader but confused by technical terms.",
        "reading_style": "Reads carefully but is cautious about unfamiliar jargon.",
        "belief_seed": (
            "If the policy uses unfamiliar technical jargon, I may treat it as unclear unless the text explains it plainly."
        ),
    },
}

PP_C_TEXT = """
Privacy Policy (Last revised: 1 April 2024)
PRIVACTY Co., Ltd. (hereinafter the “Company”) has established the following
privacy policy (hereinafter the “Policy”) regarding the handling of personal
information obtained through the step tracker application (hereinafter the “App”).
When using this App for the first time, the user shall agree to this Policy and use
this App. Users can check this Policy at any time from the settings of this App.
1. Compliance with Act on the Protection of Personal Information
The Company complies with Act on the Protection of Personal Information, guidelines
on the Act and other laws, regulations and guidelines regarding the handling of
personal data of users.
2. Data Collection and Purposes of Use
In the App, the Company uses personal information specified below for the following
purposes. Please note that if you do not provide this information, you may not be
able to use all or part of the App.
- Email address and password: To manage accounts.
- Age, gender, weight, location data, device activity data: To provide the basic
functions of the service.
- Advertisement identifier: To deliver advertisement and to measure advertisement
effectiveness.
- User action data: To understand needs for the App, to identify problems that may
occur on the App and their causes, and to develop new services.
3. How to Collect
- Provided by users: age, email address, gender, password, and weight
- Automatic collection: advertising identifier, device activity data, location data, user
action data on the App.
4. Provision of Personal Information
The Company does not provide personal information to third parties except in the
following cases:
- When we have obtained the user consent in advance
- Provision in accordance with laws and regulations
- When providing personal information to a third party without obtaining the user
consent is permitted under the Personal Information Protection Act.
(*) Note that There is no outsourcing to third-party organizations related to the
provision of the App and the Service.
However, the Company jointly uses personal information within the following
scope:
- Personal data to be jointly used: Data described in “2. Data Collection and Purposes
of Use.”
- Scope of joint users: The Company and its subsidiaries and affiliates
- Purpose of use by the joint users: To achieve purposes described in “2. Data
Collection and Purposes of Use.”
- The person responsible for the data management: PRIVACY Co., Ltd. [address]
[name of CEO]
5. Security Measures
a. Systematic Security Measures
- Establishment of a personal data manager and clarification of his/her role.
- Establishment of a reporting system in the event an incident occurs.
- Internal security audits and audits to maintain ISO27001 certification are conducted.
b. Human Security Measures
- Employees are required to submit a pledge regarding confidentiality of information.
- Continuous education on information security is provided.
c. Physical Security Measures
- Access control is implemented in areas where personal information is handled.
- Measures are taken to prevent theft or loss of devices, documents, and other items
that handle personal information.
d. Technical Security Measures
- Access control is implemented on servers and other information devices.
- A system is in place to protect against unauthorized external access and software.
- Periodic reviews of system security are conducted.
6. Stop Providing Personal Information
The App does not provide a means stop automatically providing personal data. If
you wish to stop providing personal data, please uninstall the App.
7. Inquiries
For comments, questions, complaints, or other inquiries regarding the handling of
personal information, please contact us through this inquiry form.
8. Revision of the Privacy Policy
The Company may revise the Policy from time to time, and any changes will be
posted on the App. Customers are advised to thoroughly check the latest version of
the Policy posted on the App.
""".strip()

QUESTION_DATA = {
    "id": "Q3",
    "text": "Select all options that are correct for security measures taken by PRIVACY Co., Ltd.",
    "options": {
        "A": "Measures to prevent loss of devices",
        "B": "Maintaining security certification by external organization",
        "C": "Outsourcing supervision",
        "D": "Cannot be determined",
    },
    "correct": ["A", "B"],
}

# ============================================================
# TOKEN / TEXT HELPERS
# ============================================================

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "is", "are", "was",
    "were", "be", "been", "being", "at", "from", "as", "that", "this", "these", "those", "it", "its",
    "we", "you", "they", "their", "company", "policy", "measures", "security", "taken", "select",
    "all", "options", "correct", "privacy", "privacty", "co", "ltd"
}
TECHNICAL_HINTS = {
    "iso27001", "iso 27001", "certification", "certified", "external organization",
    "third-party", "third party", "outsourcing", "supervision", "technical", "compliance"
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_answers(raw_answers: Any) -> List[str]:
    if isinstance(raw_answers, str):
        raw_answers = [raw_answers]
    if not isinstance(raw_answers, list):
        return []

    normalized = []
    seen = set()
    for item in raw_answers:
        if not isinstance(item, str):
            continue
        letter = item.strip().upper()
        if letter in {"A", "B", "C", "D"} and letter not in seen:
            normalized.append(letter)
            seen.add(letter)
    return normalized


def exact_match(pred: List[str], gold: List[str]) -> bool:
    return set(pred) == set(gold)


def option_is_technical(option_text: str) -> bool:
    low = option_text.lower()
    return any(h in low for h in TECHNICAL_HINTS)


def sectionize_policy(text: str) -> List[Dict[str, str]]:
    """
    Turn the policy into chunks the simulated reader can notice.
    This is intentionally simple and human-like: headings + bullets + lines.
    """
    chunks = []
    current_section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^[0-9]+\.", line) or re.match(r"^[a-d]\.", line.lower()):
            current_section = line
        chunks.append({"section": current_section, "text": line})
    return chunks


POLICY_CHUNKS = sectionize_policy(PP_C_TEXT)


# ============================================================
# PERSONA STATE / MEMORY STREAM
# ============================================================

@dataclass
class MemoryItem:
    text: str
    kind: str
    trial: int
    importance: float = 1.0


@dataclass
class PersonaState:
    memory_stream: List[MemoryItem] = field(default_factory=list)
    last_answers: List[List[str]] = field(default_factory=list)


def init_persona_states() -> Dict[str, PersonaState]:
    states = {}
    for name, persona in PERSONAS.items():
        states[name] = PersonaState(
            memory_stream=[
                MemoryItem(
                    text=persona["belief_seed"],
                    kind="seed",
                    trial=0,
                    importance=1.2,
                )
            ]
        )
    return states


def add_memory(state: PersonaState, text: str, kind: str, trial: int, importance: float = 1.0) -> None:
    state.memory_stream.append(MemoryItem(text=text, kind=kind, trial=trial, importance=importance))


def retrieve_memory(question: str, state: PersonaState, top_k: int = 4) -> List[str]:
    q_tokens = [t for t in tokenize(question) if t not in STOPWORDS]
    q_set = set(q_tokens)

    scored: List[Tuple[float, MemoryItem]] = []
    n = len(state.memory_stream)
    for idx, mem in enumerate(state.memory_stream):
        m_tokens = [t for t in tokenize(mem.text) if t not in STOPWORDS]
        overlap = len(q_set & set(m_tokens))

        recency = 1.0 - (n - 1 - idx) / max(1, n)  # later memories get higher score
        recency = clamp(recency, 0.0, 1.0)

        kind_bonus = {
            "seed": 0.20,
            "reflection": 0.55,
            "plan": 0.25,
            "answer": 0.15,
        }.get(mem.kind, 0.10)

        score = overlap + (0.35 * recency) + kind_bonus + (0.15 * mem.importance)
        scored.append((score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1].text for x in scored[:top_k]]


# ============================================================
# PERCEPTION / RETRIEVAL
# ============================================================

def score_chunk(question: str, chunk: Dict[str, str], persona_name: str) -> float:
    q_tokens = [t for t in tokenize(question) if t not in STOPWORDS]
    c_tokens = tokenize(chunk["text"])

    overlap = len(set(q_tokens) & set(c_tokens))
    score = float(overlap)

    lowered = chunk["text"].lower()

    if "physical security" in lowered or "prevent theft or loss" in lowered:
        score += 3.0
    if "iso27001" in lowered or "certification" in lowered:
        score += 3.2
    if "outsourcing" in lowered or "third-party" in lowered:
        score += 2.2

    if persona_name == "Expert" and ("iso27001" in lowered or "certification" in lowered):
        score += 2.0
    if persona_name != "Expert" and ("iso27001" in lowered or "certification" in lowered):
        score -= 0.8

    # Simulate skimming: the less expert the persona, the more it overweights headings and nearby text.
    if persona_name == "Human_Avg":
        if chunk["section"].startswith("5.") or chunk["section"].startswith("a.") or chunk["section"].startswith("c."):
            score += 0.7
    elif persona_name == "Non_Expert_Seniors":
        if chunk["section"].startswith("5.") or chunk["section"].startswith("c."):
            score += 0.4

    return score


def retrieve_relevant_chunks(question: str, persona_name: str, top_k: int = 4) -> List[Dict[str, str]]:
    scored = [(score_chunk(question, ch, persona_name), ch) for ch in POLICY_CHUNKS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored[:top_k]]


def mask_for_persona(text: str, persona_name: str) -> str:
    """
    Give the persona its own perception of technical terms.
    This is the key "human gap" part: experts preserve detail, non-experts get fuzzier wording.
    """
    if persona_name == "Expert":
        return text

    masked = text
    replacements = [
        ("ISO27001 certification", "a technical certification"),
        ("maintain ISO27001 certification", "maintain a technical certification"),
        ("ISO27001", "a technical code"),
        ("external organization", "outside organization"),
        ("third-party organizations", "other organizations"),
        ("third party organizations", "other organizations"),
    ]
    for old, new in replacements:
        masked = masked.replace(old, new)
        masked = masked.replace(old.lower(), new)

    # Non-experts often keep the gist, but lose confidence in technical wording.
    if persona_name == "Non_Expert_Seniors":
        masked = masked.replace("maintain", "somehow keep")
        masked = masked.replace("conducted", "done")
    return masked


def build_perceived_policy(question: str, persona_name: str, state: PersonaState) -> str:
    retrieved_policy = retrieve_relevant_chunks(question, persona_name, top_k=4)
    perceived_policy = [mask_for_persona(ch["text"], persona_name) for ch in retrieved_policy]
    retrieved_memory = retrieve_memory(question, state, top_k=3)

    mem_block = "\n".join(f"- {m}" for m in retrieved_memory)
    pol_block = "\n".join(f"- {p}" for p in perceived_policy)

    return f"Retrieved memories:\n{mem_block}\n\nPerceived policy excerpts:\n{pol_block}"


# ============================================================
# LLM STEPS: REFLECTION AND SUPPORT ASSESSMENT
# ============================================================

async def generate_reflection(
    persona_name: str,
    persona: Dict[str, Any],
    question: str,
    perceived_context: str,
    rng: random.Random,
) -> Dict[str, Any]:
    system_prompt = (
        "You are simulating a human participant reading a privacy policy. "
        "Generate a short internal reflection from the persona's perspective. "
        "Do not use outside knowledge. "
        "Return JSON with keys: Beliefs, Uncertainties, Heuristics. "
        "Keep each list short and human-like."
    )

    user_prompt = f"""
Persona:
{persona["profile"]}

Reading style:
{persona["reading_style"]}

Question:
{question}

Context:
{perceived_context}

Output JSON only:
{{
  "Beliefs": ["..."],
  "Uncertainties": ["..."],
  "Heuristics": ["..."]
}}
""".strip()

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=persona["behavior"]["temperature"],
        response_format={"type": "json_object"},
    )
    data = parse_json_object(response.choices[0].message.content)

    # Light post-processing to keep the reflection grounded.
    if not isinstance(data, dict):
        data = {}

    data.setdefault("Beliefs", [])
    data.setdefault("Uncertainties", [])
    data.setdefault("Heuristics", [])

    # Add one stochastic human-like heuristic sometimes.
    if persona_name == "Human_Avg" and rng.random() < 0.35:
        data["Heuristics"].append("Focus on the clearest-looking words and do not overthink it.")
    elif persona_name == "Non_Expert_Seniors" and rng.random() < 0.45:
        data["Heuristics"].append("If the wording feels technical, be cautious and prefer uncertainty.")

    return data


async def assess_option_support(
    persona_name: str,
    persona: Dict[str, Any],
    question: str,
    perceived_context: str,
    reflection: Dict[str, Any],
) -> Dict[str, Any]:
    system_prompt = (
        "You are simulating how a human persona judges answer choices while reading a privacy policy. "
        "You are NOT asked for the final answer yet. "
        "For each option, estimate how strongly the perceived context supports it from the persona's perspective. "
        "Return JSON with keys: Support, Evidence, Notes, Confidence. "
        "Support must be numbers from 0 to 100 for every option letter."
    )

    option_text = "\n".join(f"{k}. {v}" for k, v in QUESTION_DATA["options"].items())

    user_prompt = f"""
Persona:
{persona["profile"]}

Reading style:
{persona["reading_style"]}

Question:
{question}

Options:
{option_text}

Context:
{perceived_context}

Reflection:
{json.dumps(reflection, ensure_ascii=False)}

Output JSON only:
{{
  "Support": {{"A": 0, "B": 0, "C": 0, "D": 0}},
  "Evidence": {{"A": ["..."], "B": ["..."], "C": ["..."], "D": ["..."]}},
  "Notes": "short note",
  "Confidence": 0
}}
""".strip()

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=max(0.05, persona["behavior"]["temperature"] - 0.10),
        response_format={"type": "json_object"},
    )
    data = parse_json_object(response.choices[0].message.content)
    if not isinstance(data, dict):
        data = {}

    data.setdefault("Support", {"A": 0, "B": 0, "C": 0, "D": 0})
    data.setdefault("Evidence", {"A": [], "B": [], "C": [], "D": []})
    data.setdefault("Notes", "")
    data.setdefault("Confidence", 0)

    # Normalize and clamp support values.
    support = {}
    for letter in ["A", "B", "C", "D"]:
        try:
            support[letter] = int(clamp(float(data["Support"].get(letter, 0)), 0, 100))
        except Exception:
            support[letter] = 0
    data["Support"] = support
    return data


# ============================================================
# HUMAN-LIKE DECISION LAYER
# ============================================================

def apply_persona_biases(
    persona_name: str,
    persona: Dict[str, Any],
    support: Dict[str, int],
    reflection: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, float]:
    """
    Convert LLM support scores into persona-shaped decision scores.
    This is where the human-like variability and error patterns are introduced.
    """
    behavior = persona["behavior"]
    adjusted = {}

    uncertainty_text = " ".join(reflection.get("Uncertainties", [])).lower()
    heuristics_text = " ".join(reflection.get("Heuristics", [])).lower()

    for letter, raw in support.items():
        base = float(raw)

        option_text = QUESTION_DATA["options"][letter]
        technical = option_is_technical(option_text)

        # Technical term sensitivity.
        if technical and persona_name != "Expert":
            base -= behavior["jargon_penalty"]

        # Uncertainty in the reflection pushes cautious personas toward D.
        if letter != "D" and ("unclear" in uncertainty_text or "technical" in uncertainty_text):
            base -= 4.0 if persona_name == "Human_Avg" else 7.0

        if letter == "D":
            # D becomes more attractive when the persona feels uncertain.
            base += 10.0 if "unclear" in uncertainty_text else 0.0
            base += 6.0 if persona_name != "Expert" else 0.0
            base += 3.0 if "cautious" in heuristics_text else 0.0

        # Persona-specific reading style effects.
        if persona_name == "Human_Avg" and letter in {"A", "B"}:
            base += 2.0
        if persona_name == "Non_Expert_Seniors" and letter == "B":
            # Seniors can miss technical-certification meaning, but not always.
            base -= 3.0

        # Noise simulates human inconsistency / sampling variance.
        base += rng.gauss(0, behavior["noise"])

        adjusted[letter] = base

    return adjusted


def choose_answers(
    persona_name: str,
    persona: Dict[str, Any],
    support: Dict[str, int],
    reflection: Dict[str, Any],
    rng: random.Random,
) -> List[str]:
    behavior = persona["behavior"]
    adjusted = apply_persona_biases(persona_name, persona, support, reflection, rng)

    selected: List[str] = []
    non_d_letters = ["A", "B", "C"]

    # Independent inclusion probabilities for each option.
    for letter in non_d_letters:
        p = sigmoid((adjusted[letter] - behavior["base_threshold"]) / 8.0)

        # Make non-experts somewhat more hesitant when the option is technical.
        if letter == "B" and persona_name != "Expert":
            p *= 0.78 if persona_name == "Human_Avg" else 0.64

        # Human_Avg is sometimes swayed by obvious-looking wording.
        if letter == "A" and persona_name == "Human_Avg":
            p = clamp(p + 0.06, 0.0, 1.0)

        # Seniors often hesitate and omit one of the supported options.
        if persona_name == "Non_Expert_Seniors":
            p *= 0.92

        if rng.random() < p:
            selected.append(letter)

    # Decide on D when uncertainty is high or when nothing else was selected.
    max_non_d = max(adjusted[l] for l in non_d_letters)
    d_prob = sigmoid((behavior["d_threshold"] - max_non_d) / 7.5)

    if "unclear" in " ".join(reflection.get("Uncertainties", [])).lower():
        d_prob = clamp(d_prob + behavior["uncertainty_bias"], 0.0, 1.0)

    if not selected:
        if rng.random() < d_prob:
            selected = ["D"]
        else:
            # fallback: choose the highest supported non-D option
            best = max(non_d_letters, key=lambda l: adjusted[l])
            selected = [best]

    # Occasional over-selection: select an extra borderline option if it looks plausible.
    if len(selected) == 1 and rng.random() < behavior["overselect_bias"]:
        candidates = [l for l in non_d_letters if l not in selected]
        candidates.sort(key=lambda l: adjusted[l], reverse=True)
        if candidates and adjusted[candidates[0]] > behavior["base_threshold"] - 8:
            selected.append(candidates[0])

    # Occasional omission of a correct option.
    if len(selected) > 1 and rng.random() < behavior["omission_bias"]:
        weakest = min(selected, key=lambda l: adjusted[l])
        selected.remove(weakest)

    # D is a special-case answer: if D is selected, it usually stands alone.
    if "D" in selected and len(selected) > 1:
        # Keep D only if the persona is truly uncertain.
        if max_non_d > behavior["d_threshold"] - 4:
            selected = [x for x in selected if x != "D"]
        else:
            selected = ["D"]

    # Ensure unique and stable order.
    ordered = [x for x in ["A", "B", "C", "D"] if x in selected]
    return ordered


def build_reasoning(
    selected: List[str],
    support: Dict[str, int],
    reflection: Dict[str, Any],
) -> str:
    if selected == ["D"]:
        return "The wording feels incomplete or uncertain from this reader's perspective, so the policy does not feel fully determined."
    parts = []
    for letter in selected:
        opt = QUESTION_DATA["options"][letter]
        if letter == "A":
            parts.append("The policy clearly mentions measures to prevent loss or theft of devices.")
        elif letter == "B":
            parts.append("The policy mentions maintaining a security certification, which this reader treats as an external certification.")
        elif letter == "C":
            parts.append("The policy does not support outsourcing supervision, so this should not be chosen.")
        elif letter == "D":
            parts.append("The reader cannot determine the answer confidently from the text.")
    if not parts:
        parts.append("The reader did not find enough support in the visible text.")
    return " ".join(parts)


# ============================================================
# SIMULATION
# ============================================================

async def simulate_human_comprehension(
    persona_name: str,
    persona: Dict[str, Any],
    state: PersonaState,
    question_data: Dict[str, Any],
    rng: random.Random,
    trial: int,
) -> Dict[str, Any]:
    persona_with_behavior = {
        **persona,
        "behavior": PERSONA_BEHAVIOR[persona_name],
    }

    question = question_data["text"]
    perceived_context = build_perceived_policy(question, persona_name, state)

    reflection = await generate_reflection(
        persona_name=persona_name,
        persona=persona_with_behavior,
        question=question,
        perceived_context=perceived_context,
        rng=rng,
    )

    support_result = await assess_option_support(
        persona_name=persona_name,
        persona=persona_with_behavior,
        question=question,
        perceived_context=perceived_context,
        reflection=reflection,
    )

    support = support_result["Support"]
    selected = choose_answers(
        persona_name=persona_name,
        persona=persona_with_behavior,
        support=support,
        reflection=reflection,
        rng=rng,
    )

    reasoning = build_reasoning(selected, support, reflection)
    confidence = int(clamp(max(support.get(x, 0) for x in selected) if selected else 0, 0, 100))

    # Update memory stream to simulate a persistent persona.
    add_memory(
        state,
        text=f"Trial {trial}: asked about security measures; noticed {selected if selected else ['none']}.",
        kind="answer",
        trial=trial,
        importance=0.8,
    )
    add_memory(
        state,
        text=f"Reflection: {json.dumps(reflection, ensure_ascii=False)}",
        kind="reflection",
        trial=trial,
        importance=1.2,
    )
    add_memory(
        state,
        text=f"Support snapshot: {json.dumps(support, ensure_ascii=False)}",
        kind="plan",
        trial=trial,
        importance=0.9,
    )
    state.last_answers.append(selected)

    return {
        "Answer": selected,
        "Reasoning": reasoning,
        "Confidence": confidence,
        "Support": support,
        "Reflection": reflection,
        "SupportResult": support_result,
        "PerceivedContext": perceived_context,
    }


# ============================================================
# EVALUATION METRICS
# ============================================================

def summarize_trials(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    exact = 0
    avg_size = 0.0
    any_correct = 0
    empty = 0
    selected_counts = []

    option_stats = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ["A", "B", "C", "D"]}

    for rec in records:
        pred = set(rec["Answer"])
        gold = set(QUESTION_DATA["correct"])
        if pred == gold:
            exact += 1
        if pred & gold:
            any_correct += 1
        if not pred:
            empty += 1
        selected_counts.append(len(pred))

        for letter in ["A", "B", "C", "D"]:
            if letter in pred and letter in gold:
                option_stats[letter]["tp"] += 1
            elif letter in pred and letter not in gold:
                option_stats[letter]["fp"] += 1
            elif letter not in pred and letter in gold:
                option_stats[letter]["fn"] += 1

    avg_size = sum(selected_counts) / max(1, total)
    answer_tuples = [tuple(rec["Answer"]) for rec in records]
    most_common = {}
    for a in answer_tuples:
        most_common[a] = most_common.get(a, 0) + 1
    top_answer = max(most_common.items(), key=lambda x: x[1])[0] if most_common else ()

    return {
        "trials": total,
        "exact_accuracy": exact / max(1, total),
        "any_correct_rate": any_correct / max(1, total),
        "empty_rate": empty / max(1, total),
        "avg_selected_count": avg_size,
        "top_answer": list(top_answer),
        "option_stats": option_stats,
    }


# ============================================================
# MAIN
# ============================================================

async def run_study(trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED) -> None:
    rng = random.Random(seed)
    states = init_persona_states()
    all_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in PERSONAS.keys()}

    output_jsonl = "persona_test_result.jsonl"
    output_txt = "persona_test_result.txt"

    with open(output_jsonl, "w", encoding="utf-8") as jf, open(output_txt, "w", encoding="utf-8") as tf:
        for persona_name, persona in PERSONAS.items():
            print(f"\n--- Testing Persona: {persona_name} ---")
            tf.write(f"\n--- Testing Persona: {persona_name} ---\n")

            for trial in range(1, trials + 1):
                try:
                    res = await simulate_human_comprehension(
                        persona_name=persona_name,
                        persona=persona,
                        state=states[persona_name],
                        question_data=QUESTION_DATA,
                        rng=rng,
                        trial=trial,
                    )

                    ans = res["Answer"]
                    is_correct = exact_match(ans, QUESTION_DATA["correct"])

                    record = {
                        "persona": persona_name,
                        "trial": trial,
                        "Answer": ans,
                        "correct": is_correct,
                        "reasoning": res["Reasoning"],
                        "confidence": res["Confidence"],
                        "support": res["Support"],
                        "reflection": res["Reflection"],
                        "perceived_context": res["PerceivedContext"],
                    }
                    all_results[persona_name].append(record)

                    print(f"Trial {trial}: {ans} | Correct: {is_correct}")
                    print(f"Reasoning: {res['Reasoning'][:140]}...")
                    print(f"Confidence: {res['Confidence']}")
                    print(f"Support: {res['Support']}")

                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                    tf.write(f"Trial {trial}\n")
                    tf.write(f"Answer: {ans} | Correct: {is_correct}\n")
                    tf.write(f"Reasoning: {res['Reasoning']}\n")
                    tf.write(f"Confidence: {res['Confidence']}\n")
                    tf.write(f"Support: {json.dumps(res['Support'], ensure_ascii=False)}\n")
                    tf.write("-" * 40 + "\n")
                except Exception as e:
                    err = {
                        "persona": persona_name,
                        "trial": trial,
                        "error": str(e),
                    }
                    jf.write(json.dumps(err, ensure_ascii=False) + "\n")
                    print(f"Error on {persona_name} trial {trial}: {e}")

    print("\n--- RESULTS ANALYSIS ---")
    for persona_name, records in all_results.items():
        summary = summarize_trials(records)
        acc = 100.0 * summary["exact_accuracy"]
        any_correct = 100.0 * summary["any_correct_rate"]
        empty_rate = 100.0 * summary["empty_rate"]
        avg_count = summary["avg_selected_count"]
        top_answer = summary["top_answer"]

        print(f"Persona: {persona_name}")
        print(f"  Exact accuracy: {acc:.1f}%")
        print(f"  Any-correct rate: {any_correct:.1f}%")
        print(f"  Empty-rate: {empty_rate:.1f}%")
        print(f"  Avg selected count: {avg_count:.2f}")
        print(f"  Most common answer: {top_answer}")
        print(f"  Option stats: {summary['option_stats']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona simulation for privacy-policy comprehension.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    asyncio.run(run_study(trials=args.trials, seed=args.seed))


if __name__ == "__main__":
    main()