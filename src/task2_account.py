"""
task2_account.py
=================
Phase 4 / Task 2 — Account Health & 90-Day Ticket Filter.

Pipeline:
    1. Load tickets + accounts via parse_data.py
    2. Filter a given account's tickets to a rolling 90-day window ending at
       a reference date (pandas datetime arithmetic)
    3. Summarize ticket volume, severity, and open/resolved counts
    4. Prompt Gemini 3.6 Flash with structured JSON output
       (`response_schema=AccountHealthResponse`) to produce a health verdict
       and executive summary
    5. Print pretty JSON for 2 sample accounts

NOTE ON DATA CONTRACT
----------------------
Confirmed from task1_triage.py's earlier diagnostics:
    parse_data.load_tickets() -> pandas.DataFrame with columns:
        ticket_id, account_id, product, priority, status, subject,
        description, created_at, updated_at, satisfaction_score
    priority values look like "P1".."P4" (seen "P3" in sample data) rather
    than the Low/Medium/High/Critical scale used in task1 — this file
    treats "P1"/"P2" as the critical/high band; adjust PRIORITY_HIGH_SET
    below if your actual priority values differ.

The accounts loader's exact name wasn't confirmed the same way KB's was, so
this file uses the same auto-detecting adapter pattern as task1_triage.py:
it tries common function names and tells you exactly what to fix if none
match. Expected account record fields (dict keys or DataFrame columns):
    account_id (str), account_name / name (str)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_data  # noqa: E402  (VERIFIED WORKING module in this project)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ==========================================================================
# Config — adjust here if your data's priority scale differs
# ==========================================================================
PRIORITY_HIGH_SET = {"P1", "P2"}          # treated as "critical/high"
OPEN_STATUS_SET = {"Open", "Pending Customer", "In Progress", "New"}
RESOLVED_STATUS_SET = {"Resolved", "Closed"}


# ==========================================================================
# 1. Pydantic Response Schema
# ==========================================================================
class AccountHealthResponse(BaseModel):
    account_id: str
    account_name: str
    reference_date: str
    total_tickets_90d: int
    critical_high_tickets_90d: int
    open_tickets_count: int
    health_status: Literal["Healthy", "At-Risk", "Critical Churn Risk"]
    executive_summary: str


# ==========================================================================
# Adapter helpers (mirrors task1_triage.py's pattern)
# ==========================================================================
def _first_available(module: Any, names: List[str]):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _to_record_dict(record: Any) -> dict:
    if isinstance(record, dict):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump()
    if hasattr(record, "__dict__"):
        return dict(record.__dict__)
    raise TypeError(f"Unsupported record type: {type(record)}")


def _records_from_loader_result(raw: Any) -> List[dict]:
    if isinstance(raw, pd.DataFrame):
        return raw.to_dict("records")

    records = []
    skipped = 0
    for r in raw:
        if isinstance(r, dict):
            records.append(_to_record_dict(r))
        else:
            skipped += 1
    if skipped:
        print(
            f"[WARN] Skipped {skipped} non-dict item(s) from a loader result "
            "(e.g. stray column-name strings mixed into the list) — this "
            "usually means the underlying parse_data.py loader has a bug "
            "worth checking/fixing at the source.",
            file=sys.stderr,
        )
    return records


def _load_tickets_df() -> pd.DataFrame:
    """Returns the raw tickets DataFrame (kept as a DataFrame — not records —
    because filter_recent_tickets() needs vectorized pandas datetime ops)."""
    loader = _first_available(
        parse_data, ["load_tickets", "get_tickets", "read_tickets", "tickets"]
    )
    if loader is None:
        raise AttributeError(
            "Could not find a tickets loader in parse_data.py. Expected one "
            "of: load_tickets(), get_tickets(), read_tickets(), or a "
            "`tickets` DataFrame/list."
        )
    raw = loader() if callable(loader) else loader
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        df = pd.DataFrame(_records_from_loader_result(raw))
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def _load_accounts() -> List[dict]:
    loader = _first_available(
        parse_data,
        ["load_accounts", "load_account", "get_accounts", "accounts"],
    )
    if loader is None:
        raise AttributeError(
            "Could not find an accounts loader in parse_data.py. Expected "
            "one of: load_accounts(), get_accounts(), or an `accounts` "
            "list/DataFrame. Please point _load_accounts() at the correct "
            "name in task2_account.py."
        )
    raw = loader() if callable(loader) else loader
    return _records_from_loader_result(raw)


def _field(record: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        if k in record and record[k] is not None and str(record[k]) != "nan":
            return str(record[k])
    return default


# ==========================================================================
# 2. 90-Day Ticket Filtering Function
# ==========================================================================
def filter_recent_tickets(
    tickets_df: pd.DataFrame,
    account_id: str,
    reference_date: Union[str, datetime],
) -> pd.DataFrame:
    """
    Returns the subset of tickets_df belonging to `account_id`, created in
    the 90-day window (reference_date - 90 days, reference_date], inclusive
    of the reference date.
    """
    if isinstance(reference_date, str):
        ref_dt = pd.to_datetime(reference_date, utc=True)
    else:
        ref_dt = pd.to_datetime(reference_date)
        if ref_dt.tzinfo is None:
            ref_dt = ref_dt.tz_localize("UTC")

    window_start = ref_dt - timedelta(days=90)

    mask = (
        (tickets_df["account_id"] == account_id)
        & (tickets_df["created_at"] >= window_start)
        & (tickets_df["created_at"] <= ref_dt)
    )
    return tickets_df.loc[mask].copy()


# ==========================================================================
# 3. Account Health Evaluator Function
# ==========================================================================
def _get_genai_client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. Add it to your .env "
            "file (see .env.example)."
        )
    return genai.Client(api_key=api_key)


def _summarize_tickets(recent_tickets: pd.DataFrame) -> dict:
    total = int(len(recent_tickets))

    if "priority" in recent_tickets.columns:
        crit_high = int(recent_tickets["priority"].isin(PRIORITY_HIGH_SET).sum())
        priority_counts = recent_tickets["priority"].value_counts().to_dict()
    else:
        crit_high = 0
        priority_counts = {}

    if "status" in recent_tickets.columns:
        open_count = int(recent_tickets["status"].isin(OPEN_STATUS_SET).sum())
        status_counts = recent_tickets["status"].value_counts().to_dict()
    else:
        open_count = 0
        status_counts = {}

    return {
        "total": total,
        "critical_high": crit_high,
        "open_count": open_count,
        "priority_counts": priority_counts,
        "status_counts": status_counts,
    }


def _build_prompt(
    account_id: str,
    account_name: str,
    reference_date: str,
    stats: dict,
    recent_tickets: pd.DataFrame,
    account_record: Optional[dict] = None,
) -> str:
    ticket_lines = []
    for _, row in recent_tickets.head(15).iterrows():
        ticket_lines.append(
            f"- [{row.get('ticket_id', '?')}] priority={row.get('priority', '?')} "
            f"status={row.get('status', '?')} subject=\"{row.get('subject', '')}\""
        )
    tickets_block = "\n".join(ticket_lines) or "No tickets in the 90-day window."

    account_context_block = "No additional account record available."
    if account_record:
        escalation_notes = account_record.get("escalation_notes") or []
        escalation_str = (
            "; ".join(str(n) for n in escalation_notes) if escalation_notes else "None"
        )
        account_context_block = (
            f"- Plan tier: {account_record.get('plan_tier', 'unknown')}\n"
            f"- ARR (USD): {account_record.get('arr_usd', 'unknown')}\n"
            f"- Seats active/licensed: {account_record.get('seats_active', '?')}/"
            f"{account_record.get('seats_licensed', '?')}\n"
            f"- Usage trend: {account_record.get('usage_trend', 'unknown')}\n"
            f"- Last login (days ago): {account_record.get('last_login_days_ago', 'unknown')}\n"
            f"- NPS score: {account_record.get('nps_score', 'unknown')}\n"
            f"- Open tickets (all-time, per account record): {account_record.get('open_tickets', 'unknown')}\n"
            f"- P1 tickets last 30d (per account record): {account_record.get('p1_tickets_last_30d', 'unknown')}\n"
            f"- Renewal date: {account_record.get('renewal_date', 'unknown')}\n"
            f"- Last QBR date: {account_record.get('last_qbr_date', 'unknown')}\n"
            f"- Existing CS-labeled health status (for reference only — form your "
            f"own judgment, don't just copy this): {account_record.get('health_status', 'unknown')}\n"
            f"- Escalation notes: {escalation_str}"
        )

    return f"""You are a customer success analyst assessing account health for a B2B SaaS company.

ACCOUNT: {account_name} ({account_id})
REFERENCE DATE: {reference_date}

ACCOUNT RECORD CONTEXT:
{account_context_block}

90-DAY TICKET WINDOW STATS (computed from ticket data, authoritative):
- Total tickets: {stats['total']}
- Critical/High priority tickets: {stats['critical_high']}
- Open tickets: {stats['open_count']}
- Priority breakdown: {stats['priority_counts']}
- Status breakdown: {stats['status_counts']}

RECENT TICKETS (up to 15 shown):
{tickets_block}

Instructions:
1. Set "account_id" to "{account_id}", "account_name" to "{account_name}",
   "reference_date" to "{reference_date}".
2. Set "total_tickets_90d" to {stats['total']}, "critical_high_tickets_90d" to
   {stats['critical_high']}, "open_tickets_count" to {stats['open_count']}.
3. Classify "health_status" as one of: Healthy, At-Risk, Critical Churn Risk.
   Weigh BOTH the 90-day ticket trend/severity AND the account record context
   (usage trend, login recency, NPS, escalation notes, renewal proximity).
   Explicit churn signals in escalation notes or a declining usage trend
   should push toward "At-Risk" or "Critical Churn Risk" even if raw ticket
   volume looks modest.
4. Write a concise 2-4 sentence "executive_summary" a customer success
   manager could read in 10 seconds — call out the single strongest risk or
   health signal (ticket trend, escalation note, usage/NPS, renewal timing),
   not just a restatement of the numbers.

Respond ONLY with a JSON object matching the required schema — no extra text."""


def evaluate_account_health(
    account_id: str,
    tickets_df: pd.DataFrame,
    accounts: List[dict],
    reference_date: Union[str, datetime],
    model: str = "gemini-3.6-flash",
) -> AccountHealthResponse:
    account_record = next(
        (a for a in accounts if _field(a, "account_id", "id") == account_id), None
    )
    account_name = (
        _field(account_record, "account_name", "company", "name", default=account_id)
        if account_record
        else account_id
    )

    if isinstance(reference_date, datetime):
        ref_date_str = reference_date.date().isoformat()
    else:
        ref_date_str = str(reference_date)[:10]

    recent_tickets = filter_recent_tickets(tickets_df, account_id, reference_date)
    stats = _summarize_tickets(recent_tickets)

    prompt = _build_prompt(
        account_id, account_name, ref_date_str, stats, recent_tickets, account_record
    )

    client = _get_genai_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": AccountHealthResponse,
        },
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, AccountHealthResponse):
        result = parsed
    else:
        data = json.loads(response.text)
        result = AccountHealthResponse(**data)

    # Guarantee the numeric/identity fields match ground truth even if the
    # model drifted on them — these are computed facts, not judgment calls.
    result.account_id = account_id
    result.account_name = account_name
    result.reference_date = ref_date_str
    result.total_tickets_90d = stats["total"]
    result.critical_high_tickets_90d = stats["critical_high"]
    result.open_tickets_count = stats["open_count"]

    return result


# ==========================================================================
# 4. Execution Test
# ==========================================================================
if __name__ == "__main__":
    try:
        tickets_df = _load_tickets_df()
    except AttributeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        accounts = _load_accounts()
    except AttributeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if tickets_df.empty or not accounts:
        print("[ERROR] No tickets or accounts loaded — nothing to evaluate.")
        sys.exit(1)

    # Reference date = latest ticket creation date in the dataset
    reference_date = tickets_df["created_at"].max()
    print(f"Using reference date: {reference_date.date().isoformat()}\n")

    sample_account_ids = (
        tickets_df["account_id"].dropna().unique().tolist()
    )
    known_account_ids = {_field(a, "account_id", "id") for a in accounts}
    valid_sample_ids = [aid for aid in sample_account_ids if aid in known_account_ids][:2]

    if not valid_sample_ids:
        print(
            "[WARN] None of the sample ticket account_ids exist in the "
            "accounts table — falling back to the first 2 known accounts "
            "instead so the demo still runs against real account context."
        )
        valid_sample_ids = list(known_account_ids)[:2]

    print(f"Evaluating health for {len(valid_sample_ids)} account(s)...\n")

    for acc_id in valid_sample_ids:
        try:
            result = evaluate_account_health(acc_id, tickets_df, accounts, reference_date)
            print(f"--- Account {acc_id} ---")
            print(json.dumps(result.model_dump(), indent=2))
            print()
        except Exception as e:
            print(f"--- Account {acc_id} FAILED ---")
            print(f"Error: {e}\n")
