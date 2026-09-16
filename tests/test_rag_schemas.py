import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from scripts.validate_style_rag_data import validate_dataset
from tools.rag.schemas import (
    SourceAdmissionPolicy,
    SourceRecord,
    StyleCard,
    StyleQueryManifest,
    compute_style_card_hash,
    production_admission,
)


ROOT = Path(__file__).resolve().parents[1]
CARDS_PATH = ROOT / "data/miaoxiang/style_templates.v2.jsonl"
SOURCES_PATH = ROOT / "data/miaoxiang/source_catalog.jsonl"
ADMISSION_AUDIT_PATH = ROOT / "data/miaoxiang/style_templates.v2.admission-audit.json"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RagSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = [StyleCard.model_validate(row) for row in _load_jsonl(CARDS_PATH)]
        cls.sources = [SourceRecord.model_validate(row) for row in _load_jsonl(SOURCES_PATH)]
        cls.sources_by_id = {source.source_id: source for source in cls.sources}

    def test_current_cards_parse_and_hash(self) -> None:
        self.assertEqual(len(self.cards), 17)
        self.assertTrue(all(compute_style_card_hash(card) == card.content_sha256 for card in self.cards))

    def test_current_sources_parse(self) -> None:
        self.assertEqual(len(self.sources), 7)
        self.assertEqual(len(self.sources_by_id), 7)

    def test_source_admission_policy_is_fail_closed(self) -> None:
        payload = json.loads(ADMISSION_AUDIT_PATH.read_text(encoding="utf-8"))
        policy = SourceAdmissionPolicy.model_validate(payload["production_source_admission"])
        self.assertEqual(policy.default_status, "fixed_query_only")
        self.assertEqual(policy.cards["style-edm-trance-v2"].status, "fixed_query_only")
        self.assertEqual(
            policy.cards["style-edm-drum-and-bass-v2"].status,
            "fixed_query_only",
        )

    def test_holdout_manifest_parse_and_query_labels(self) -> None:
        manifest_path = ROOT / "data/miaoxiang/style_fixed_queries.v1.3.json"
        manifest = StyleQueryManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        self.assertEqual(len(manifest.queries), 20)
        self.assertEqual(len({query.query_id for query in manifest.queries}), 20)

    def test_role_scope_and_composite_boundary(self) -> None:
        composite = next(card for card in self.cards if card.card_role == "composite_query_template")
        self.assertEqual(composite.retrieval_scope, "query_expansion_only")
        eligible, reasons = production_admission(composite, self.sources_by_id[composite.source_id])
        self.assertFalse(eligible)
        self.assertIn("review_status_not_approved", reasons)
        self.assertIn("not_a_direct_style_candidate", reasons)
        self.assertIn("source_admission_not_granted", reasons)

    def test_draft_cards_are_not_production_eligible(self) -> None:
        for card in self.cards:
            eligible, reasons = production_admission(card, self.sources_by_id[card.source_id])
            self.assertFalse(eligible)
            self.assertIn("review_status_not_approved", reasons)
            self.assertIn("source_admission_not_granted", reasons)

    def test_approved_direct_card_requires_explicit_source_admission(self) -> None:
        card = self.cards[0].model_copy(update={"review_status": "approved"})
        eligible, reasons = production_admission(card, self.sources_by_id[card.source_id])
        self.assertFalse(eligible)
        self.assertEqual(reasons, ["source_admission_not_granted"])
        eligible, reasons = production_admission(
            card,
            self.sources_by_id[card.source_id],
            source_admission_status="production_derivative_allowed",
        )
        self.assertTrue(eligible)
        self.assertEqual(reasons, [])

    def test_invalid_voice_partial_fields_are_rejected(self) -> None:
        payload = _load_jsonl(CARDS_PATH)[0]
        payload.pop("default_language")
        with self.assertRaises(ValidationError):
            StyleCard.model_validate(payload)

    def test_invalid_hash_is_rejected_by_admission(self) -> None:
        card = self.cards[0].model_copy(update={"content_sha256": "0" * 64})
        eligible, reasons = production_admission(card, self.sources_by_id[card.source_id])
        self.assertFalse(eligible)
        self.assertIn("content_sha256_mismatch", reasons)

    def test_dry_run_has_no_admitted_cards_and_no_mutation_flags(self) -> None:
        report = validate_dataset(CARDS_PATH, SOURCES_PATH)
        self.assertEqual(report["card_count"], 17)
        self.assertEqual(report["source_count"], 7)
        self.assertEqual(report["valid_card_count"], 17)
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["production_eligible_count"], 0)
        self.assertFalse(report["review_status_changed"])
        self.assertFalse(report["production_index_changed"])
        self.assertEqual(report["blocked_reason_counts"]["review_status_not_approved"], 17)
        self.assertEqual(report["blocked_reason_counts"]["source_admission_not_granted"], 17)
        self.assertEqual(report["source_admission_default_status"], "fixed_query_only")

    def test_dry_run_reports_unknown_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cards_path = Path(directory) / "cards.jsonl"
            payload = _load_jsonl(CARDS_PATH)[0]
            payload["source_id"] = "missing-source"
            cards_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            report = validate_dataset(cards_path, SOURCES_PATH)
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["card_results"][0]["errors"], ["source_id_not_in_catalog"])
        self.assertFalse(report["card_results"][0]["production_eligible"])

    def test_missing_admission_audit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_audit = Path(directory) / "missing-admission-audit.json"
            report = validate_dataset(CARDS_PATH, SOURCES_PATH, missing_audit)
        self.assertEqual(report["admission_policy_errors"], ["FileNotFoundError"])
        self.assertEqual(report["source_admission_default_status"], "blocked")
        self.assertEqual(report["production_eligible_count"], 0)
        self.assertEqual(report["blocked_reason_counts"]["source_admission_not_granted"], 17)


if __name__ == "__main__":
    unittest.main()
