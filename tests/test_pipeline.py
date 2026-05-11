import unittest

from pipeline import (
    evaluate_extractions,
    load_dataset,
    run_pipeline,
    validate_schema,
)


class PipelineTests(unittest.TestCase):
    def test_validate_schema_requires_all_fields(self):
        with self.assertRaises(ValueError):
            validate_schema({"ticket_id": "T-1"})

    def test_pipeline_outputs_valid_entities(self):
        dataset = load_dataset("data/customer_support_dataset.json")
        results = run_pipeline(dataset)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].issue_type, "refund_request")
        self.assertEqual(results[1].requested_action, "check_shipment")
        self.assertEqual(results[2].product, "subscription")

    def test_metrics_include_core_scores(self):
        dataset = load_dataset("data/customer_support_dataset.json")
        predictions = run_pipeline(dataset)
        metrics = evaluate_extractions(predictions, [item["gold_entities"] for item in dataset])

        self.assertIn("exact_match_accuracy", metrics)
        self.assertIn("micro_f1", metrics)
        self.assertGreaterEqual(metrics["micro_f1"], 0.0)
        self.assertLessEqual(metrics["micro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
