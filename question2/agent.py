
from __future__ import annotations

import json
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket
from question1.agent import parse_resolution


@dataclass
class TicketResolution:
    action: str
    kb_article: str


@dataclass
class TriageResult:
    action: str | None
    kb_article: str
    reason: str


class SupportPipeline:
    """Four-stage ShopAssist pipeline with one LLM call per ticket."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        # Node 1: Triage
        triage = self._triage(thread, order)

        # Node 2: Retrieve
        article = self._retrieve(triage.kb_article)

        # Node 3: Compose (the only LLM call)
        draft = self._compose(
            ticket_id, thread, snapshot, triage, article
        )

        # Node 4: Audit
        return self._audit(draft, triage)

    def _triage(self, thread: str, order: dict) -> TriageResult:
        """Identify the applicable policy from ticket and order facts."""

        product_type = order.get("product_type", "")
        delivered_days = order.get("delivered_days_ago", 0)
        damaged = order.get("damage_reported", False)
        customized = order.get("customized", False)
        opened = order.get("opened", False)
        license_revealed = order.get("license_revealed", False)

        if product_type == "digital" and license_revealed:
            return TriageResult(
                "SEND_ARTICLE",
                "digital-license-no-refund",
                "The digital license has already been revealed.",
            )

        if customized and damaged:
            return TriageResult(
                "ESCALATE_TIER2",
                "custom-items-no-replace",
                "Customized damaged merchandise requires manual review.",
            )

        if damaged and not customized and delivered_days <= 7:
            return TriageResult(
                "REPLACE",
                "damage-replacement-7-days",
                "A standard item was damaged within seven days.",
            )

        if (
            product_type == "physical"
            and not opened
            and delivered_days <= 30
        ):
            return TriageResult(
                "REFUND",
                "returns-within-30-days",
                "The unopened physical item is within the 30-day return window.",
            )

        if product_type == "subscription":
            return TriageResult(
                "SEND_ARTICLE",
                "subscription-cooling-off",
                "Subscription refund eligibility depends on cancellation age.",
            )

        return TriageResult(
            None,
            "returns-within-30-days",
            "Further policy validation is required.",
        )

    def _retrieve(self, article_id: str) -> str:
        """Load only the relevant article, not the expanded help center."""
        return load_article(article_id)

    def _compose(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        triage: TriageResult,
        article: str,
    ) -> str:
        messages = [
            ChatMessage(
                "system",
                "You are ShopAssist. Use the order facts and supplied policy "
                "article. Return only valid JSON with action, kb_article, "
                "and customer_reply. Keep the reply concise.",
            ),
            ChatMessage(
                "user",
                f"TICKET: {ticket_id}\n"
                f"THREAD:\n{thread}\n"
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT\n"
                f"TRIAGE: {triage.reason}\n"
                f"CANDIDATE_ACTION: {triage.action}\n"
                f"ARTICLE_ID: {triage.kb_article}\n"
                f"POLICY:\n{article}",
            ),
        ]

        return self.llm.chat("compose", messages)

    def _audit(
        self,
        draft: str,
        triage: TriageResult,
    ) -> TicketResolution:
        """Validate the model output against the locally retrieved policy."""

        try:
            parsed = parse_resolution(draft)
            action = parsed.action
            article_id = parsed.kb_article
        except (ValueError, KeyError, TypeError):
            action = None
            article_id = None

        valid_actions = {
            "REFUND",
            "REPLACE",
            "SEND_ARTICLE",
            "ESCALATE_TIER2",
        }

        # Reject invalid model output or a decision that conflicts
        # with the policy identified during triage.
        if (
            action not in valid_actions
            or article_id != triage.kb_article
            or (
                triage.action is not None
                and action != triage.action
            )
        ):
            if triage.action is None:
                raise ValueError(
                    "Audit failed: no valid model decision or triage fallback."
                )

            action = triage.action
            article_id = triage.kb_article

        return TicketResolution(action, article_id)
