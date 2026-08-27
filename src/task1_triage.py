"""
task1_triage.py
================
Phase 3 / Task 1 — Ticket Triage & RAG.

Pipeline:
    1. Load knowledge-base (KB) articles via parse_data.py
    2. Index KB content with TF-IDF (sklearn), with a pure-Python keyword
       fallback if scikit-learn isn't installed
    3. For each incoming ticket, retrieve the top-k most relevant KB articles
    4. Send the ticket + retrieved context to Gemini 2.5 Flash and force
       structured JSON output matching the `TicketTriageResponse` schema
    5. Print pretty JSON results for the first N tickets

NOTE ON DATA CONTRACT
----------------------
This module expects `parse_data.py` to expose loader functions returning
lists of dict-like or attribute-like records. Since `parse_data.py` is
already verified working in this project but its exact function names
weren't specified in the task brief, this file uses a small adapter layer
(`_load_tickets`, `_load_kb_articles`) that tries several common naming
conventions (`load_tickets`, `get_tickets`, `TICKETS`, etc.) before failing
with a clear error telling you exactly what to wire up. If your actual
function names differ, just point the two TODOs near the top of those
functions at the real names — everything downstream is agnostic to the
loader's exact signature as long as it yields ticket/KB records with the
fields below.

Expected ticket record fields (dict keys or attributes):
    ticket_id (str), subject (str, optional), description (str)

Expected KB record fields (dict keys or attributes):
    kb_id (str), title (str, optional), content (str)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Make sure `src/` is importable regardless of CWD the script is run from.
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_data  # noqa: E402  (VERIFIED WORKING module in this project)

try:
    from dotenv import load_dotenv

    load_dotenv()  # reads .env in the project root, if present
except ImportError:
    pass  # python-dotenv not installed; env vars must be set some other way


# ==========================================================================
# 1. Pydantic Response Schema
# ==========================================================================
class TicketTriageResponse(BaseModel):
    ticket_id: str
    priority: Literal["Low", "Medium", "High", "Critical"]
    category: Literal["Billing", "Technical", "Account", "Feature Request"]
    retrieved_kb_ids: List[str] = Field(default_factory=list)
    draft_response: str


# ==========================================================================
# Small adapter helpers to bridge to parse_data.py without assuming an
# exact function name. Edit the candidate lists below if needed.
# ==========================================================================
def _first_available(module: Any, names: List[str]):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _to_record_dict(record: Any) -> dict:
    """Normalize a dict-like, object-like, or raw-string record to a plain dict."""
    if isinstance(record, dict):
        return record
    if hasattr(record, "model_dump"):  # pydantic model
        return record.model_dump()
    if hasattr(record, "__dict__"):
        return dict(record.__dict__)
    if isinstance(record, str):
        # Loader returned raw strings instead of structured records.
        # Try JSON first (e.g. each line/item is a JSON-encoded record).
        try:
            parsed = json.loads(record)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        # Otherwise assume it's a plain ticket ID with no separate text yet —
        # this WILL produce empty descriptions until parse_data.py's real
        # contract is confirmed. Flagged loudly so it's not silently wrong.
        print(
            f"[WARN] Got a bare string record ('{record[:60]}...' if long) "
            "instead of a dict — parse_data.py's loader likely needs a "
            "different call. Wrapping as ticket_id only for now.",
            file=sys.stderr,
        )
        return {"ticket_id": record, "description": ""}
    raise TypeError(f"Unsupported record type: {type(record)}")


def _records_from_loader_result(raw: Any) -> List[dict]:
    """Normalize a loader's return value — DataFrame, list of dicts/objects,
    or list of strings — into a list of plain dicts."""
    # pandas DataFrame (this is what parse_data.py actually returns)
    try:
        import pandas as pd

        if isinstance(raw, pd.DataFrame):
            return raw.to_dict("records")
    except ImportError:
        pass

    return [_to_record_dict(r) for r in raw]


def _load_tickets() -> List[dict]:
    # TODO: adjust candidate names if parse_data.py uses something else
    loader = _first_available(
        parse_data, ["load_tickets", "get_tickets", "read_tickets", "tickets"]
    )
    if loader is None:
        raise AttributeError(
            "Could not find a tickets loader in parse_data.py. "
            "Expected one of: load_tickets(), get_tickets(), read_tickets(), "
            "or a `tickets` list/constant. Please point _load_tickets() at "
            "the correct name."
        )
    raw = loader() if callable(loader) else loader
    return _records_from_loader_result(raw)


def _load_kb_articles() -> List[dict]:
    # TODO: adjust candidate names if parse_data.py uses something else
    loader = _first_available(
        parse_data,
        ["load_kb_articles", "load_kb", "load_knowledge_base", "get_kb_articles", "kb_articles"],
    )
    if loader is None:
        raise AttributeError(
            "Could not find a KB loader in parse_data.py. Expected one of: "
            "load_kb_articles(), load_knowledge_base(), get_kb_articles(), "
            "or a `kb_articles` list/constant. Please point _load_kb_articles() "
            "at the correct name."
        )
    raw = loader() if callable(loader) else loader
    return _records_from_loader_result(raw)


def _field(record: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        if k in record and record[k] is not None:
            return str(record[k])
    return default


# ==========================================================================
# 2. RAG / KB Retriever Class
# ==========================================================================
class KBRetriever:
    """
    TF-IDF based retriever over KB article content, with a pure-Python
    keyword-overlap fallback when scikit-learn is unavailable.
    """

    def __init__(self, kb_articles: Optional[List[dict]] = None):
        self.kb_articles: List[dict] = kb_articles if kb_articles is not None else _load_kb_articles()
        self.kb_ids: List[str] = [
            _field(a, "kb_id", "doc_id", "id", "filename", default=f"KB-{i}")
            for i, a in enumerate(self.kb_articles)
        ]
        self.kb_texts: List[str] = [
            (_field(a, "title", default="") + " " + _field(a, "content", "text", "body", default="")).strip()
            for a in self.kb_articles
        ]

        self._use_sklearn = False
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401

            self._TfidfVectorizer = TfidfVectorizer
            self._cosine_similarity = cosine_similarity
            self._use_sklearn = True
        except ImportError:
            print(
                "[KBRetriever] scikit-learn not found — falling back to pure "
                "Python keyword-overlap similarity.",
                file=sys.stderr,
            )

        if self._use_sklearn and self.kb_texts:
            self._vectorizer = self._TfidfVectorizer(stop_words="english")
            self._kb_matrix = self._vectorizer.fit_transform(self.kb_texts)
        else:
            self._vectorizer = None
            self._kb_matrix = None

    # ---- pure python fallback -------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> set:
        return {w.strip(".,!?;:\"'()[]").lower() for w in text.split() if len(w) > 2}

    def _keyword_similarity(self, query: str) -> List[float]:
        query_tokens = self._tokenize(query)
        scores = []
        for doc_text in self.kb_texts:
            doc_tokens = self._tokenize(doc_text)
            if not query_tokens or not doc_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & doc_tokens)
            score = overlap / (len(query_tokens) ** 0.5 * len(doc_tokens) ** 0.5)
            scores.append(score)
        return scores

    # ---- public API -------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 2) -> List[dict]:
        """
        Returns up to top_k dicts: {"kb_id": str, "content": str, "score": float}
        sorted by descending relevance.
        """
        if not self.kb_texts:
            return []

        if self._use_sklearn:
            query_vec = self._vectorizer.transform([query])
            sims = self._cosine_similarity(query_vec, self._kb_matrix)[0]
        else:
            sims = self._keyword_similarity(query)

        ranked = sorted(
            range(len(sims)), key=lambda i: sims[i], reverse=True
        )[:top_k]

        return [
            {
                "kb_id": self.kb_ids[i],
                "content": self.kb_texts[i],
                "score": float(sims[i]),
            }
            for i in ranked
            if sims[i] > 0
        ] or [
            # if everything scored 0, still return the top_k so triage has
            # *some* context rather than none
            {"kb_id": self.kb_ids[i], "content": self.kb_texts[i], "score": float(sims[i])}
            for i in ranked
        ]


# ==========================================================================
# 3. LLM Triage Function
# ==========================================================================
def _get_genai_client():
    """
    Initializes the google-genai client. Expects GEMINI_API_KEY (or
    GOOGLE_API_KEY) to be set in the environment / .env file.
    """
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. Add it to your .env "
            "file (see .env.example) and load it, e.g. via python-dotenv."
        )
    return genai.Client(api_key=api_key)


def _build_prompt(ticket_id: str, ticket_text: str, kb_context: List[dict]) -> str:
    context_block = "\n\n".join(
        f"[KB {c['kb_id']}]\n{c['content']}" for c in kb_context
    ) or "No relevant knowledge base articles were found."

    return f"""You are a support-ticket triage assistant for an enterprise helpdesk.

TICKET ID: {ticket_id}
TICKET DESCRIPTION:
{ticket_text}

RELEVANT KNOWLEDGE BASE ARTICLES:
{context_block}

Instructions:
1. Classify the ticket's "priority" as one of: Low, Medium, High, Critical.
2. Classify the ticket's "category" as one of: Billing, Technical, Account, Feature Request.
3. List the KB article IDs you actually used to ground your response in "retrieved_kb_ids"
   (use the IDs exactly as given above, e.g. "{kb_context[0]['kb_id'] if kb_context else 'KB-001'}").
4. Write a concise, professional, and helpful "draft_response" that a support
   agent could send to the customer, grounded in the KB content where relevant.

Respond ONLY with a JSON object matching the required schema — no extra text."""


def triage_ticket(ticket: dict, retriever: KBRetriever, model: str = "gemini-3.6-flash") -> TicketTriageResponse:
    """
    Runs retrieval + Gemini structured-output triage for a single ticket dict.
    """
    ticket_id = _field(ticket, "ticket_id", "id", default="UNKNOWN")
    subject = _field(ticket, "subject", default="")
    description = _field(ticket, "description", "body", "text", default="")
    ticket_text = f"{subject}\n{description}".strip()

    kb_context = retriever.retrieve(ticket_text, top_k=2)
    prompt = _build_prompt(ticket_id, ticket_text, kb_context)

    client = _get_genai_client()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": TicketTriageResponse,
        },
    )

    # google-genai exposes the schema-validated object via `.parsed` when
    # response_schema is a Pydantic model; fall back to manual parsing
    # of `.text` for older SDK versions just in case.
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, TicketTriageResponse):
        result = parsed
    else:
        data = json.loads(response.text)
        data.setdefault("ticket_id", ticket_id)
        result = TicketTriageResponse(**data)

    # Guarantee ticket_id integrity even if the model altered it
    if result.ticket_id != ticket_id:
        result.ticket_id = ticket_id

    return result


# ==========================================================================
# 4. Testing Logic
# ==========================================================================
if __name__ == "__main__":
    try:
        tickets = _load_tickets()
    except AttributeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not tickets:
        print("[ERROR] No tickets loaded from parse_data.py — nothing to triage.")
        sys.exit(1)

    retriever = KBRetriever()

    sample_tickets = tickets[:2]
    print(f"Running triage on {len(sample_tickets)} ticket(s)...\n")

    for t in sample_tickets:
        tid = _field(t, "ticket_id", "id", default="UNKNOWN")
        try:
            result = triage_ticket(t, retriever)
            print(f"--- Ticket {tid} ---")
            print(json.dumps(result.model_dump(), indent=2))
            print()
        except Exception as e:
            print(f"--- Ticket {tid} FAILED ---")
            print(f"Error: {e}\n")
