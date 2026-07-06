from __future__ import annotations

from src.etl_extractor import ETLExtractor


def test_rule_based_extractor_finds_equipment_actions_and_category() -> None:
    extractor = ETLExtractor(mode="rule_based")

    fields = extractor.extract(
        "Responded to high vibration on pump P-104. Found bearing wear. "
        "Root cause: inadequate lubrication. Replaced mechanical seal and bearing set."
    )

    assert fields.equipment_tag == "P-104"
    assert fields.failure_mode == "bearing wear"
    assert fields.parts_replaced is not None
    assert "mechanical seal" in fields.parts_replaced
    assert fields.root_cause == "inadequate lubrication"
    assert fields.failure_category == "mechanical_failure"
    assert fields.extractor_used == "rule_based"


def test_rule_based_extractor_keeps_vague_text_low_content() -> None:
    extractor = ETLExtractor(mode="rule_based")

    fields = extractor.extract("Checked unit after general note from operations. No clear fault listed.")

    assert fields.equipment_tag is None
    assert fields.failure_mode is None
    assert fields.parts_replaced is None
    assert fields.root_cause is None
    assert fields.failure_category is None
    assert fields.confidence <= 0.15
    assert fields.extractor_used == "rule_based"
