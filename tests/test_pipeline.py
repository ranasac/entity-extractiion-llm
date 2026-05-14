import pytest

from pipeline import (
    evaluate_extractions,
    get_llm_utility,
    HeuristicLLMUtility,
    load_dataset,
    run_pipeline,
    validate_schema,
)


def test_validate_schema_requires_all_fields():
    with pytest.raises(ValueError):
        validate_schema({"ticket_id": "T-1"})


def test_pipeline_outputs_valid_entities():
    dataset = load_dataset("data/customer_support_dataset.json")
    results = run_pipeline(dataset, llm=HeuristicLLMUtility())
    assert len(results) == 3
    assert results[0].issue_type == "refund_request"
    assert results[1].requested_action == "check_shipment"
    assert results[2].product == "subscription"


def test_metrics_include_core_scores():
    dataset = load_dataset("data/customer_support_dataset.json")
    predictions = run_pipeline(dataset, llm=HeuristicLLMUtility())
    metrics = evaluate_extractions(predictions, [item["gold_entities"] for item in dataset])

    assert "exact_match_accuracy" in metrics
    assert "micro_f1" in metrics
    assert 0.0 <= metrics["micro_f1"] <= 1.0


def test_get_llm_utility_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(get_llm_utility(), HeuristicLLMUtility)
