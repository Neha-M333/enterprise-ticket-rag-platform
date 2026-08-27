"""
eval_harness.py
================
Phase 5 / Task 3 — Automated Evaluation Harness for Task 1 (Ticket Triage/RAG)
and Task 2 (Account Health), including adversarial test cases.

Reuses the real, verified functions/classes from task1_triage.py and
task2_account.py rather than reimplementing any logic:
    - task1_triage.KBRetriever, task1_triage.triage_ticket,
      task1_triage.TicketTriageResponse
    - task2_account.filter_recent_tickets, task2_account.evaluate_account_health,
      task2_account.AccountHealthResponse, task2_account._summarize_tickets

Runs 10 test cases total (5 per task), scores each with a PASS/FAIL verdict
plus a 0.0-1.0 quality score, prints a summary table to the console, and
writes the full report to eval_report.json in the project root.

Run from the project root:
    python src/eval_harness.py

For quota-free smoke-testing of the harness itself (fixture loading, 90-day
math, JSON report, summary table) without calling Gemini at all, use:
    python src/eval_harness.py --mock
Mock mode fabricates deterministic TicketTriageResponse / AccountHealthResponse
objects instead of hitting the API, and every result derived from a mock call
is clearly labeled MOCK in the report so it's never mistaken for a real,
API-graded result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_data                      # noqa: E402  (VERIFIED WORKING)
import task1_triage                    # noqa: E402  (VERIFIED WORKING)
import task2_account                   # noqa: E402  (VERIFIED WORKING)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ==========================================================================
# Result data structures
# ==========================================================================
@dataclass
class EvalResult:
    test_id: str
    task: str
    name: str
    status: str            # "PASS", "FAIL", or "SKIPPED" (API quota exhausted)
    quality_score: float   # 0.0 - 1.0
    details: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class EvalHarness:
    def __init__(self, mock: bool = False):
        self.mock = mock
        self.results: List[EvalResult] = []

        # Shared fixtures, loaded once
        self.tickets: List[dict] = []
        self.tickets_df: pd.DataFrame = pd.DataFrame()
        self.accounts: List[dict] = []
        self.kb_articles: List[dict] = []
        self.retriever: Optional[task1_triage.KBRetriever] = None
        self.reference_date = None

        # Cache LLM calls by key so tests that examine different facets of
        # the same triage/health call don't each burn a separate API request
        # against the (often tiny) free-tier daily quota.
        self._triage_cache: dict = {}
        self._health_cache: dict = {}

    # ---------------------------------------------------------------
    # Fixture loading
    # ---------------------------------------------------------------
    def load_fixtures(self) -> None:
        self.tickets = task1_triage._load_tickets()
        self.tickets_df = task2_account._load_tickets_df()
        self.accounts = task2_account._load_accounts()
        self.kb_articles = task1_triage._load_kb_articles()
        self.retriever = task1_triage.KBRetriever(self.kb_articles)

        if not self.tickets_df.empty and "created_at" in self.tickets_df.columns:
            self.reference_date = self.tickets_df["created_at"].max()
        else:
            self.reference_date = pd.Timestamp.now(tz="UTC")

    # ---------------------------------------------------------------
    # Helper: run one test safely, recording pass/fail/skip + score
    # ---------------------------------------------------------------
    @staticmethod
    def _is_quota_error(err_str: str) -> bool:
        return "RESOURCE_EXHAUSTED" in err_str or "429" in err_str

    def _run(
        self,
        test_id: str,
        task: str,
        name: str,
        fn: Callable[[], "tuple[bool, float, str]"],
    ) -> None:
        try:
            passed, score, details = fn()
            self.results.append(
                EvalResult(
                    test_id=test_id,
                    task=task,
                    name=name,
                    status="PASS" if passed else "FAIL",
                    quality_score=round(max(0.0, min(1.0, score)), 2),
                    details=details,
                )
            )
        except Exception as e:  # noqa: BLE001 - harness must never crash on a bad test
            err_str = f"{type(e).__name__}: {e}"
            is_quota = self._is_quota_error(err_str)
            self.results.append(
                EvalResult(
                    test_id=test_id,
                    task=task,
                    name=name,
                    status="SKIPPED" if is_quota else "FAIL",
                    quality_score=0.0,
                    details=(
                        "Skipped — Gemini API quota exhausted (see error)."
                        if is_quota
                        else "Unhandled exception during test execution."
                    ),
                    error=err_str,
                )
            )

    # ---------------------------------------------------------------
    # LLM call wrapper: a daily quota (as opposed to a per-minute rate
    # limit) cannot be fixed by sleeping within the same run, so retries
    # default to 0. Pass max_retries=1+ explicitly only if you have reason
    # to believe the 429 is a short-lived rate limit, not a daily cap.
    # ---------------------------------------------------------------
    def _call_with_backoff(self, fn: Callable[[], Any], max_retries: int = 0):
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if not self._is_quota_error(msg) or attempt == max_retries:
                    raise
                last_err = e
                match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", msg)
                delay = (int(match.group(1)) if match else 30) + 2
                delay = min(delay, 90)
                print(
                    f"  [retry] Quota hit — waiting {delay}s before retrying...",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise last_err  # pragma: no cover

    # ---------------------------------------------------------------
    # Mock fixtures — used only in --mock mode, so the harness's own
    # logic (fixture loading, filtering, scoring, reporting) can be
    # smoke-tested with zero Gemini calls.
    # ---------------------------------------------------------------
    @staticmethod
    def _mock_triage_response(ticket: dict) -> "task1_triage.TicketTriageResponse":
        tid = ticket.get("ticket_id", "UNKNOWN")
        text = f"{ticket.get('subject', '')} {ticket.get('description', '')}".lower()
        priority = "High" if any(w in text for w in ["broken", "down", "urgent", "fail"]) else "Medium"
        return task1_triage.TicketTriageResponse(
            ticket_id=tid,
            priority=priority,
            category="Technical",
            retrieved_kb_ids=["MOCK-KB-1"],
            draft_response=(
                "[MOCK] Thanks for reaching out — we've reviewed your request and "
                "our team is looking into it. We'll follow up shortly with next steps."
            ),
        )

    @staticmethod
    def _mock_health_response(account_id: str, account_name: str, ref_date_str: str, stats: dict) -> "task2_account.AccountHealthResponse":
        return task2_account.AccountHealthResponse(
            account_id=account_id,
            account_name=account_name,
            reference_date=ref_date_str,
            total_tickets_90d=stats["total"],
            critical_high_tickets_90d=stats["critical_high"],
            open_tickets_count=stats["open_count"],
            health_status="At-Risk" if stats["critical_high"] > 0 else "Healthy",
            executive_summary=(
                f"[MOCK] Account had {stats['total']} tickets in the last 90 days "
                f"with {stats['critical_high']} critical/high priority and "
                f"{stats['open_count']} still open."
            ),
        )

    def _cached_triage(self, ticket: dict) -> "task1_triage.TicketTriageResponse":
        tid = ticket.get("ticket_id", "UNKNOWN")
        if tid not in self._triage_cache:
            if self.mock:
                self._triage_cache[tid] = self._mock_triage_response(ticket)
            else:
                self._triage_cache[tid] = self._call_with_backoff(
                    lambda: task1_triage.triage_ticket(ticket, self.retriever)
                )
        return self._triage_cache[tid]

    def _cached_health(self, account_id: str) -> "task2_account.AccountHealthResponse":
        if account_id not in self._health_cache:
            if self.mock:
                account_record = next(
                    (a for a in self.accounts if task2_account._field(a, "account_id", "id") == account_id),
                    None,
                )
                account_name = (
                    task2_account._field(account_record, "account_name", "company", "name", default=account_id)
                    if account_record
                    else account_id
                )
                ref_date_str = str(self.reference_date)[:10]
                recent = task2_account.filter_recent_tickets(self.tickets_df, account_id, self.reference_date)
                stats = task2_account._summarize_tickets(recent)
                self._health_cache[account_id] = self._mock_health_response(
                    account_id, account_name, ref_date_str, stats
                )
            else:
                self._health_cache[account_id] = self._call_with_backoff(
                    lambda: task2_account.evaluate_account_health(
                        account_id, self.tickets_df, self.accounts, self.reference_date
                    )
                )
        return self._health_cache[account_id]

    def _pick_ticket(self, keyword: Optional[str] = None) -> dict:
        """Grab a representative ticket dict from the loaded fixture set."""
        if not self.tickets:
            raise RuntimeError("No tickets available in fixtures.")
        if keyword:
            for t in self.tickets:
                text = f"{t.get('subject', '')} {t.get('description', '')}".lower()
                if keyword.lower() in text:
                    return t
        return self.tickets[0]

    def _pick_known_account_id(self) -> str:
        if not self.accounts:
            raise RuntimeError("No accounts available in fixtures.")
        ticket_account_ids = self.tickets_df["account_id"].dropna().unique().tolist()
        account_ids = {task2_account._field(a, "account_id", "id") for a in self.accounts}
        for aid in ticket_account_ids:
            if aid in account_ids:
                return aid
        return task2_account._field(self.accounts[0], "account_id", "id")

    # ==================================================================
    # TASK 1 TEST SUITE
    # ==================================================================
    def test1_1_valid_classification(self):
        def run():
            ticket = self._pick_ticket()
            result = self._cached_triage(ticket)
            valid_priority = result.priority in {"Low", "Medium", "High", "Critical"}
            valid_category = result.category in {"Billing", "Technical", "Account", "Feature Request"}
            passed = valid_priority and valid_category
            score = 1.0 if passed else 0.0
            details = (
                f"ticket_id={result.ticket_id} priority={result.priority} "
                f"category={result.category}"
            )
            return passed, score, details

        self._run("T1-1", "Task1", "Valid Classification", run)

    def test1_2_kb_retrieval(self):
        def run():
            ticket = self._pick_ticket()
            ticket_text = f"{ticket.get('subject', '')} {ticket.get('description', '')}"
            retrieved = self.retriever.retrieve(ticket_text, top_k=2)
            passed = len(retrieved) > 0
            score = 1.0 if passed else 0.0
            kb_ids = [r["kb_id"] for r in retrieved]
            details = f"retrieved_kb_ids={kb_ids}"
            return passed, score, details

        self._run("T1-2", "Task1", "KB Retrieval Relevance", run)

    def test1_3_draft_response_non_empty(self):
        def run():
            ticket = self._pick_ticket()
            result = self._cached_triage(ticket)
            length = len(result.draft_response.strip())
            passed = length > 50
            score = min(1.0, length / 150) if passed else 0.0
            details = f"draft_response_length={length}"
            return passed, score, details

        self._run("T1-3", "Task1", "Draft Response Non-Empty", run)

    def test1_4_deterministic_schema(self):
        def run():
            ticket = self._pick_ticket()
            result = self._cached_triage(ticket)
            raw_json = json.dumps(result.model_dump())
            reparsed = task1_triage.TicketTriageResponse(**json.loads(raw_json))
            passed = reparsed.model_dump() == result.model_dump()
            score = 1.0 if passed else 0.0
            details = "Round-tripped JSON matches the original TicketTriageResponse."
            return passed, score, details

        self._run("T1-4", "Task1", "Deterministic Schema Round-Trip", run)

    def test1_5_adversarial_ticket(self):
        def run():
            adversarial_ticket = {
                "ticket_id": "ADV-0001",
                "subject": "",
                "description": "It is broken please fix",
            }
            try:
                if self.mock:
                    result = self._mock_triage_response(adversarial_ticket)
                else:
                    result = self._call_with_backoff(
                        lambda: task1_triage.triage_ticket(adversarial_ticket, self.retriever)
                    )
            except Exception as e:
                raise  # let _run() classify quota errors as SKIPPED vs FAIL

            valid_priority = result.priority in {"Low", "Medium", "High", "Critical"}
            valid_category = result.category in {"Billing", "Technical", "Account", "Feature Request"}
            has_response = len(result.draft_response.strip()) > 0
            passed = valid_priority and valid_category and has_response
            score = 1.0 if passed else 0.5
            details = (
                f"No exception raised. priority={result.priority} "
                f"category={result.category} response_len={len(result.draft_response)}"
            )
            return passed, score, details

        self._run("T1-5", "Task1", "Adversarial Ambiguous Ticket", run)

    # ==================================================================
    # TASK 2 TEST SUITE
    # ==================================================================
    def test2_1_ninety_day_math(self):
        def run():
            synthetic = pd.DataFrame(
                [
                    {"ticket_id": "S-1", "account_id": "ACC-TEST", "priority": "P1",
                     "status": "Open", "subject": "in-window",
                     "created_at": pd.Timestamp("2026-06-01", tz="UTC")},
                    {"ticket_id": "S-2", "account_id": "ACC-TEST", "priority": "P3",
                     "status": "Resolved", "subject": "in-window-edge",
                     "created_at": pd.Timestamp("2026-04-02", tz="UTC")},
                    {"ticket_id": "S-3", "account_id": "ACC-TEST", "priority": "P2",
                     "status": "Closed", "subject": "outside-window",
                     "created_at": pd.Timestamp("2026-01-01", tz="UTC")},
                ]
            )
            reference_date = pd.Timestamp("2026-07-01", tz="UTC")
            filtered = task2_account.filter_recent_tickets(synthetic, "ACC-TEST", reference_date)
            ids_in = set(filtered["ticket_id"].tolist())
            passed = ids_in == {"S-1", "S-2"}
            score = 1.0 if passed else 0.0
            details = f"included_ticket_ids={sorted(ids_in)} (expected only S-1, S-2)"
            return passed, score, details

        self._run("T2-1", "Task2", "90-Day Window Math", run)

    def test2_2_metric_accuracy(self):
        def run():
            synthetic = pd.DataFrame(
                [
                    {"ticket_id": "S-1", "account_id": "ACC-TEST", "priority": "P1",
                     "status": "Open", "subject": "a",
                     "created_at": pd.Timestamp("2026-06-01", tz="UTC")},
                    {"ticket_id": "S-2", "account_id": "ACC-TEST", "priority": "P3",
                     "status": "Resolved", "subject": "b",
                     "created_at": pd.Timestamp("2026-06-10", tz="UTC")},
                    {"ticket_id": "S-3", "account_id": "ACC-TEST", "priority": "P2",
                     "status": "Open", "subject": "c",
                     "created_at": pd.Timestamp("2026-06-15", tz="UTC")},
                ]
            )
            reference_date = pd.Timestamp("2026-07-01", tz="UTC")
            filtered = task2_account.filter_recent_tickets(synthetic, "ACC-TEST", reference_date)
            stats = task2_account._summarize_tickets(filtered)

            expected_total = 3
            expected_crit_high = 2   # P1 + P2 fall in PRIORITY_HIGH_SET
            expected_open = 2        # two rows with status == "Open"

            passed = (
                stats["total"] == expected_total
                and stats["critical_high"] == expected_crit_high
                and stats["open_count"] == expected_open
            )
            score = 1.0 if passed else 0.0
            details = f"stats={stats}"
            return passed, score, details

        self._run("T2-2", "Task2", "Metric Accuracy vs Manual Filter", run)

    def test2_3_health_status_validity(self):
        def run():
            account_id = self._pick_known_account_id()
            result = self._cached_health(account_id)
            allowed = {"Healthy", "At-Risk", "Critical Churn Risk"}
            passed = result.health_status in allowed
            score = 1.0 if passed else 0.0
            details = f"account_id={account_id} health_status={result.health_status}"
            return passed, score, details

        self._run("T2-3", "Task2", "Health Status Enum Validity", run)

    def test2_4_executive_summary_quality(self):
        def run():
            account_id = self._pick_known_account_id()
            result = self._cached_health(account_id)
            summary = result.executive_summary
            mentions_total = str(result.total_tickets_90d) in summary
            mentions_relevant_word = any(
                w.lower() in summary.lower()
                for w in ["ticket", "risk", "renewal", "usage", "health", "escalat"]
            )
            length_ok = len(summary.strip()) > 30
            passed = length_ok and mentions_relevant_word
            score = (
                0.5 * float(length_ok)
                + 0.3 * float(mentions_relevant_word)
                + 0.2 * float(mentions_total)
            )
            details = f"executive_summary={summary!r}"
            return passed, score, details

        self._run("T2-4", "Task2", "Executive Summary Quality", run)

    def test2_5_adversarial_account(self):
        def run():
            fake_account_id = "ACC-DOES-NOT-EXIST-9999"
            if self.mock:
                recent = task2_account.filter_recent_tickets(
                    self.tickets_df, fake_account_id, self.reference_date
                )
                stats = task2_account._summarize_tickets(recent)
                result = self._mock_health_response(
                    fake_account_id, fake_account_id, str(self.reference_date)[:10], stats
                )
            else:
                result = self._call_with_backoff(
                    lambda: task2_account.evaluate_account_health(
                        fake_account_id, self.tickets_df, self.accounts, self.reference_date
                    )
                )

            zero_tickets_handled = result.total_tickets_90d == 0
            valid_status = result.health_status in {"Healthy", "At-Risk", "Critical Churn Risk"}
            passed = zero_tickets_handled and valid_status
            score = 1.0 if passed else 0.5
            details = (
                f"No exception raised for unknown account. "
                f"total_tickets_90d={result.total_tickets_90d} "
                f"health_status={result.health_status} "
                f"account_name_fallback={result.account_name}"
            )
            return passed, score, details

        self._run("T2-5", "Task2", "Adversarial Unknown Account", run)

    # ==================================================================
    # Orchestration
    # ==================================================================
    def run_all(self) -> None:
        self.load_fixtures()

        self.test1_1_valid_classification()
        self.test1_2_kb_retrieval()
        self.test1_3_draft_response_non_empty()
        self.test1_4_deterministic_schema()
        self.test1_5_adversarial_ticket()

        self.test2_1_ninety_day_math()
        self.test2_2_metric_accuracy()
        self.test2_3_health_status_validity()
        self.test2_4_executive_summary_quality()
        self.test2_5_adversarial_account()

    # ---------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------
    def print_summary_table(self) -> None:
        if self.mock:
            print("*** MOCK MODE — no Gemini API calls were made; LLM-backed results are fabricated. ***\n")
        widths = {"id": 6, "task": 7, "name": 34, "status": 6, "score": 6}
        header = (
            f"{'ID':<{widths['id']}} {'Task':<{widths['task']}} "
            f"{'Test Name':<{widths['name']}} {'Status':<{widths['status']}} "
            f"{'Score':<{widths['score']}}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for r in self.results:
            print(
                f"{r.test_id:<{widths['id']}} {r.task:<{widths['task']}} "
                f"{r.name:<{widths['name']}} {r.status:<{widths['status']}} "
                f"{r.quality_score:<{widths['score']}}"
            )
        print(sep)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed_n = sum(1 for r in self.results if r.status == "FAIL")
        skipped_n = sum(1 for r in self.results if r.status == "SKIPPED")
        graded = total - skipped_n
        avg_score = (
            sum(r.quality_score for r in self.results if r.status != "SKIPPED") / graded
            if graded
            else 0.0
        )
        print(f"Passed: {passed}/{total}   Failed: {failed_n}   Skipped (quota): {skipped_n}")
        print(f"Average Quality Score (excl. skipped): {avg_score:.2f}")
        print(sep)

        failed = [r for r in self.results if r.status == "FAIL"]
        skipped = [r for r in self.results if r.status == "SKIPPED"]
        if failed:
            print("\nFailed test details:")
            for r in failed:
                print(f"  [{r.test_id}] {r.name}: {r.details}")
                if r.error:
                    print(f"      error: {r.error}")
        if skipped:
            print("\nSkipped (Gemini API quota exhausted) — rerun later to grade these:")
            for r in skipped:
                print(f"  [{r.test_id}] {r.name}")

    def save_report(self, path: str) -> None:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed_n = sum(1 for r in self.results if r.status == "FAIL")
        skipped_n = sum(1 for r in self.results if r.status == "SKIPPED")
        graded = total - skipped_n
        avg_score = (
            sum(r.quality_score for r in self.results if r.status != "SKIPPED") / graded
            if graded
            else 0.0
        )

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mock_mode": self.mock,
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed_n,
                "skipped_quota": skipped_n,
                "average_quality_score_excl_skipped": round(avg_score, 3),
            },
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved full evaluation report to: {path}")


try:
    import pytest as _pytest
except ImportError:
    _pytest = None


if _pytest is not None:
    # ======================================================================
    # Optional pytest integration
    # ======================================================================
    # This lets `pytest src/eval_harness.py -v` collect and report the same
    # 10 checks individually through standard pytest tooling, in addition to
    # the harness's own console table + eval_report.json (produced by
    # `python src/eval_harness.py`, see main() below).
    #
    # The harness runs exactly ONCE per pytest session (module-scoped
    # fixture) — each test_* function below just asserts on the
    # already-computed result for its test_id, so pytest never multiplies
    # the number of live Gemini calls.
    @_pytest.fixture(scope="module")
    def _harness():
        h = EvalHarness()
        h.run_all()
        return h

    def _assert_result(harness: "EvalHarness", test_id: str) -> None:
        result = next((r for r in harness.results if r.test_id == test_id), None)
        assert result is not None, f"No result recorded for {test_id}"
        if result.status == "SKIPPED":
            _pytest.skip(result.details)
        assert result.status == "PASS", (
            f"{result.name}: {result.details}"
            + (f" | {result.error}" if result.error else "")
        )

    def test_T1_1_valid_classification(_harness):
        _assert_result(_harness, "T1-1")

    def test_T1_2_kb_retrieval(_harness):
        _assert_result(_harness, "T1-2")

    def test_T1_3_draft_response_non_empty(_harness):
        _assert_result(_harness, "T1-3")

    def test_T1_4_deterministic_schema(_harness):
        _assert_result(_harness, "T1-4")

    def test_T1_5_adversarial_ticket(_harness):
        _assert_result(_harness, "T1-5")

    def test_T2_1_ninety_day_math(_harness):
        _assert_result(_harness, "T2-1")

    def test_T2_2_metric_accuracy(_harness):
        _assert_result(_harness, "T2-2")

    def test_T2_3_health_status_validity(_harness):
        _assert_result(_harness, "T2-3")

    def test_T2_4_executive_summary_quality(_harness):
        _assert_result(_harness, "T2-4")

    def test_T2_5_adversarial_account(_harness):
        _assert_result(_harness, "T2-5")


def main():
    parser = argparse.ArgumentParser(description="Automated evaluation harness for Task 1 + Task 2.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without calling the Gemini API — fabricates deterministic "
             "responses so the harness's own logic can be validated for free.",
    )
    args = parser.parse_args()

    harness = EvalHarness(mock=args.mock)
    try:
        harness.run_all()
    except Exception as e:
        print(f"[FATAL] Could not initialize fixtures / run harness: {e}")
        traceback.print_exc()
        sys.exit(1)

    harness.print_summary_table()

    report_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_report.json")
    )
    harness.save_report(report_path)


if __name__ == "__main__":
    main()