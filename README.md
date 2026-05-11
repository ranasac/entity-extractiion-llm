# entity-extraction-llm

This repository includes an end-to-end applied exercise for **prompt engineering + entity extraction** on customer support text.

## What is included
- Fake customer support dataset: `data/customer_support_dataset.json`
- LLM utility simulation + prompt template: `pipeline.py` (`FakeLLMUtility` + `build_prompt`)
- Schema enforcement and validation: `validate_schema`
- End-to-end run + evaluation metrics: `run_pipeline` and `evaluate_extractions`
- Focused tests: `tests/test_pipeline.py`

## Run
```bash
python pipeline.py
```

## Test
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Prompt engineering approach
The prompt constrains output to JSON only, with explicit required fields and allowed values for categorical fields (`issue_type`, `priority`, `sentiment`).

## Schema enforcement and validation
`validate_schema` enforces:
- required keys
- no unexpected keys
- non-empty strings for every field
- enum checks for `issue_type`, `priority`, and `sentiment`

Invalid responses raise `ValueError`.

## Metrics used
`evaluate_extractions` reports:
- `exact_match_accuracy`: full-row match rate across all extracted fields
- `field_accuracy`: per-field correctness
- `micro_precision`, `micro_recall`, `micro_f1`: aggregate field-level extraction quality
