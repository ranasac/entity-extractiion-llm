import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Union


REQUIRED_FIELDS = {
    "ticket_id",
    "customer_name",
    "email",
    "order_id",
    "issue_type",
    "product",
    "priority",
    "sentiment",
    "requested_action",
}

ALLOWED_ISSUE_TYPES = {
    "refund_request",
    "delivery_issue",
    "technical_issue",
    "billing_issue",
    "account_access",
}
ALLOWED_PRIORITIES = {"low", "medium", "high"}
ALLOWED_SENTIMENTS = {"negative", "neutral", "positive"}


@dataclass
class ExtractionResult:
    ticket_id: str
    customer_name: str
    email: str
    order_id: str
    issue_type: str
    product: str
    priority: str
    sentiment: str
    requested_action: str


class LLMUtility(Protocol):
    def extract(self, ticket_id: str, ticket_text: str) -> str:
        ...


def build_prompt(ticket_text: str) -> str:
    return (
        "You are an information extraction assistant for customer support tickets. "
        "Return ONLY compact JSON with these string fields: "
        "ticket_id, customer_name, email, order_id, issue_type, product, priority, sentiment, requested_action. "
        "Use issue_type in [refund_request, delivery_issue, technical_issue, billing_issue, account_access], "
        "priority in [low, medium, high], sentiment in [negative, neutral, positive]. "
        "Ticket:\n"
        f"{ticket_text}"
    )


def validate_schema(payload: Dict[str, str]) -> ExtractionResult:
    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    extra = payload.keys() - REQUIRED_FIELDS
    if extra:
        raise ValueError(f"Unexpected fields: {sorted(extra)}")

    for key in REQUIRED_FIELDS:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field '{key}' must be a non-empty string")

    if payload["issue_type"] not in ALLOWED_ISSUE_TYPES:
        raise ValueError("Invalid issue_type")
    if payload["priority"] not in ALLOWED_PRIORITIES:
        raise ValueError("Invalid priority")
    if payload["sentiment"] not in ALLOWED_SENTIMENTS:
        raise ValueError("Invalid sentiment")

    return ExtractionResult(**payload)


class LangChainLLMUtility:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain-openai is required for LangChainLLMUtility. "
                "Install dependencies from requirements.txt."
            ) from exc
        self._llm = ChatOpenAI(model=model, temperature=0)

    def extract(self, ticket_id: str, ticket_text: str) -> str:
        prompt = build_prompt(ticket_text)
        response = self._llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        payload = _parse_llm_json(content)
        payload["ticket_id"] = ticket_id
        return json.dumps(payload)


class HeuristicLLMUtility:
    def extract(self, ticket_id: str, ticket_text: str) -> str:
        prompt = build_prompt(ticket_text)
        _ = prompt  # included to represent prompt-engineering usage

        customer_name = self._extract_name(ticket_text)
        email = self._extract_email(ticket_text)
        order_id = self._extract_order_id(ticket_text)
        issue_type = self._classify_issue(ticket_text)
        product = self._extract_product(ticket_text)
        priority = self._classify_priority(ticket_text)
        sentiment = self._classify_sentiment(ticket_text)
        requested_action = self._requested_action(issue_type)

        return json.dumps(
            {
                "ticket_id": ticket_id,
                "customer_name": customer_name,
                "email": email,
                "order_id": order_id,
                "issue_type": issue_type,
                "product": product,
                "priority": priority,
                "sentiment": sentiment,
                "requested_action": requested_action,
            }
        )

    @staticmethod
    def _extract_name(text: str) -> str:
        patterns = [r"(?:I'm|I am|This is) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)"]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return "Unknown Customer"

    @staticmethod
    def _extract_email(text: str) -> str:
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        return match.group(0) if match else "unknown@example.com"

    @staticmethod
    def _extract_order_id(text: str) -> str:
        match = re.search(r"ORD-\d+", text)
        return match.group(0) if match else "ORD-UNKNOWN"

    @staticmethod
    def _extract_product(text: str) -> str:
        lower = text.lower()
        if "router" in lower:
            return "router"
        if "laptop" in lower:
            return "laptop"
        if "subscription" in lower:
            return "subscription"
        return "general"

    @staticmethod
    def _classify_issue(text: str) -> str:
        lower = text.lower()
        if "refund" in lower:
            return "refund_request"
        if "late" in lower or "delivery" in lower or "shipment" in lower:
            return "delivery_issue"
        if "can't login" in lower or "password" in lower or "locked" in lower:
            return "account_access"
        if "bill" in lower or "charged" in lower or "invoice" in lower:
            return "billing_issue"
        return "technical_issue"

    @staticmethod
    def _classify_priority(text: str) -> str:
        lower = text.lower()
        if "urgent" in lower or "asap" in lower or "today" in lower:
            return "high"
        if "when possible" in lower or "soon" in lower:
            return "medium"
        return "low"

    @staticmethod
    def _classify_sentiment(text: str) -> str:
        lower = text.lower()
        if any(token in lower for token in ["frustrated", "angry", "disappointed", "not working"]):
            return "negative"
        if any(token in lower for token in ["thanks", "thank you", "appreciate"]):
            return "positive"
        return "neutral"

    @staticmethod
    def _requested_action(issue_type: str) -> str:
        mapping = {
            "refund_request": "process_refund",
            "delivery_issue": "check_shipment",
            "technical_issue": "provide_technical_support",
            "billing_issue": "review_billing",
            "account_access": "reset_account_access",
        }
        return mapping[issue_type]


def load_dataset(path: Union[str, Path]) -> List[Dict[str, object]]:
    dataset_path = Path(path)
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def run_pipeline(
    dataset: List[Dict[str, object]],
    llm: Union[LLMUtility, None] = None,
) -> List[ExtractionResult]:
    extractor = llm or get_llm_utility()
    results: List[ExtractionResult] = []
    for item in dataset:
        raw_json = extractor.extract(item["ticket_id"], item["text"])
        payload = json.loads(raw_json)
        results.append(validate_schema(payload))
    return results


def evaluate_extractions(
    predictions: List[ExtractionResult],
    ground_truth: List[Dict[str, str]],
) -> Dict[str, object]:
    fields = [field for field in REQUIRED_FIELDS if field != "ticket_id"]
    field_correct = {field: 0 for field in fields}
    total_fields = len(fields) * len(predictions)
    exact_matches = 0

    for predicted, gold in zip(predictions, ground_truth):
        row_match = True
        for field in fields:
            if getattr(predicted, field) == gold[field]:
                field_correct[field] += 1
            else:
                row_match = False
        if row_match:
            exact_matches += 1

    tp = 0
    fp = 0
    fn = 0
    for predicted, gold in zip(predictions, ground_truth):
        for field in fields:
            predicted_value = getattr(predicted, field)
            gold_value = gold[field]
            if predicted_value == gold_value:
                tp += 1
                continue

            if not _is_abstention(predicted_value):
                fp += 1
            if not _is_abstention(gold_value):
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "exact_match_accuracy": exact_matches / len(predictions) if predictions else 0.0,
        "field_accuracy": {
            field: field_correct[field] / len(predictions) if predictions else 0.0
            for field in fields
        },
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def _is_abstention(value: str) -> bool:
    lower = value.strip().lower()
    return lower in {"unknown", "ord-unknown", "unknown@example.com"}


def _parse_llm_json(content: str) -> Dict[str, str]:
    try:
        payload = json.loads(content)
        return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response does not contain valid JSON object")
    return json.loads(match.group(0))


def get_llm_utility() -> LLMUtility:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return LangChainLLMUtility()
    return HeuristicLLMUtility()


def main() -> None:
    dataset_path = Path("data/customer_support_dataset.json")
    dataset = load_dataset(dataset_path)
    predictions = run_pipeline(dataset)
    ground_truth = [item["gold_entities"] for item in dataset]
    metrics = evaluate_extractions(predictions, ground_truth)

    print("Predictions:")
    for prediction in predictions:
        print(json.dumps(prediction.__dict__, ensure_ascii=False))

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
