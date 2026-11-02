"""
Tests for HEDIS Care Gap Closure Agent
Tests cover: gap identification logic, risk stratification, outreach drafting,
analytics calculation, and full end-to-end orchestration.
"""
import unittest
from datetime import date, timedelta
from main import (
    PatientProfileLoader,
    CareGapIdentifier,
    RiskStratifier,
    OutreachDraftingAgent,
    AnalyticsReporter,
    HEDISCareGapOrchestrator,
    ElephantAgent,
    HEDISComplianceGoldfish,
    ClinicalPriorityGoldfish,
    OutreachClarityGoldfish,
    EGMHEDISOrchestrator,
    HEDIS_MEASURES,
)


# ============================================================
# Helpers
# ============================================================
def _patient(age=55, sex="male", conditions=None, screenings=None, risk_factors=None, **kw):
    return {
        "patient_id": "TESTPT",
        "name": "Test Patient",
        "age": age,
        "sex": sex,
        "conditions": conditions or [],
        "screenings": screenings or {},
        "risk_factors": risk_factors or {},
        "preferred_contact": "portal",
        "language": "English",
        "payer": "Commercial",
        **kw,
    }


def _gap(measure, severity="due", tier="HIGH", score=0.80):
    return {
        "measure": measure,
        "reason": f"Test gap {measure}",
        "severity": severity,
        "tier": tier,
        "priority_score": score,
        "measure_name": HEDIS_MEASURES.get(measure, {}).get("name", measure),
        "clinical_rationale": HEDIS_MEASURES.get(measure, {}).get("clinical_rationale", ""),
    }


# ============================================================
# PatientProfileLoader tests
# ============================================================
class TestPatientProfileLoader(unittest.TestCase):
    def setUp(self):
        self.loader = PatientProfileLoader()

    def test_loads_correct_size(self):
        pts = self.loader.load_synthetic_panel(15)
        self.assertEqual(len(pts), 15)

    def test_single_patient(self):
        pts = self.loader.load_synthetic_panel(1)
        self.assertEqual(len(pts), 1)

    def test_required_fields_present(self):
        pts = self.loader.load_synthetic_panel(5)
        required = {"patient_id", "name", "age", "sex", "conditions", "screenings",
                    "risk_factors", "preferred_contact", "language", "payer"}
        for p in pts:
            for field in required:
                self.assertIn(field, p, f"Missing field: {field}")

    def test_age_in_valid_range(self):
        pts = self.loader.load_synthetic_panel(30)
        for p in pts:
            self.assertGreaterEqual(p["age"], 18)
            self.assertLessEqual(p["age"], 85)

    def test_sex_valid(self):
        pts = self.loader.load_synthetic_panel(20)
        for p in pts:
            self.assertIn(p["sex"], ("male", "female"))

    def test_deterministic_with_same_seed(self):
        """Two runs with same size should produce same first patient."""
        pts1 = self.loader.load_synthetic_panel(10)
        pts2 = self.loader.load_synthetic_panel(10)
        self.assertEqual(pts1[0]["patient_id"], pts2[0]["patient_id"])
        self.assertEqual(pts1[0]["age"], pts2[0]["age"])

    def test_patient_id_unique(self):
        pts = self.loader.load_synthetic_panel(25)
        ids = [p["patient_id"] for p in pts]
        self.assertEqual(len(ids), len(set(ids)))

    def test_conditions_is_list(self):
        pts = self.loader.load_synthetic_panel(10)
        for p in pts:
            self.assertIsInstance(p["conditions"], list)

    def test_preferred_contact_valid(self):
        pts = self.loader.load_synthetic_panel(20)
        valid = {"portal", "phone", "text"}
        for p in pts:
            self.assertIn(p["preferred_contact"], valid)


# ============================================================
# CareGapIdentifier tests
# ============================================================
class TestCareGapIdentifier(unittest.TestCase):
    def setUp(self):
        self.ident = CareGapIdentifier()

    # ── Colorectal (COLO) ──────────────────────────────────────────────
    def test_colo_gap_no_colonoscopy(self):
        p = _patient(age=55, screenings={"last_colonoscopy": None})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("COLO", [g["measure"] for g in gaps])

    def test_colo_gap_colonoscopy_too_old(self):
        old = (date.today() - timedelta(days=365 * 12)).isoformat()
        p = _patient(age=55, screenings={"last_colonoscopy": old})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("COLO", [g["measure"] for g in gaps])

    def test_no_colo_gap_recent(self):
        recent = (date.today() - timedelta(days=365 * 3)).isoformat()
        p = _patient(age=55, screenings={"last_colonoscopy": recent})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("COLO", [g["measure"] for g in gaps])

    def test_no_colo_gap_age_44(self):
        p = _patient(age=44, screenings={"last_colonoscopy": None})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("COLO", [g["measure"] for g in gaps])

    def test_no_colo_gap_age_76(self):
        p = _patient(age=76, screenings={"last_colonoscopy": None})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("COLO", [g["measure"] for g in gaps])

    # ── Mammogram (BCS) ──────────────────────────────────────────────
    def test_bcs_gap_female_50_no_mammogram(self):
        p = _patient(age=55, sex="female", screenings={"last_mammogram": None})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("BCS", [g["measure"] for g in gaps])

    def test_no_bcs_gap_male(self):
        p = _patient(age=55, sex="male", screenings={"last_mammogram": None})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("BCS", [g["measure"] for g in gaps])

    def test_no_bcs_gap_recent_mammogram(self):
        recent = (date.today() - timedelta(days=180)).isoformat()
        p = _patient(age=60, sex="female", screenings={"last_mammogram": recent})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("BCS", [g["measure"] for g in gaps])

    def test_no_bcs_gap_age_49(self):
        p = _patient(age=49, sex="female", screenings={"last_mammogram": None})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("BCS", [g["measure"] for g in gaps])

    # ── Cervical (CCS) ───────────────────────────────────────────────
    def test_ccs_gap_female_no_pap(self):
        p = _patient(age=35, sex="female", screenings={"last_pap_smear": None})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("CCS", [g["measure"] for g in gaps])

    def test_no_ccs_gap_male(self):
        p = _patient(age=35, sex="male", screenings={"last_pap_smear": None})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("CCS", [g["measure"] for g in gaps])

    # ── Hypertension (CBP) ──────────────────────────────────────────
    def test_cbp_gap_htn_uncontrolled(self):
        p = _patient(age=55, conditions=["hypertension"], screenings={"bp_controlled": False})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("CBP", [g["measure"] for g in gaps])

    def test_no_cbp_gap_htn_controlled(self):
        p = _patient(age=55, conditions=["hypertension"], screenings={"bp_controlled": True})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("CBP", [g["measure"] for g in gaps])

    def test_no_cbp_gap_no_htn(self):
        p = _patient(age=55, conditions=[], screenings={"bp_controlled": False})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("CBP", [g["measure"] for g in gaps])

    # ── Diabetes A1c (CDC_A1C) ──────────────────────────────────────
    def test_a1c_gap_diabetic_no_test(self):
        p = _patient(age=55, conditions=["diabetes"], screenings={"last_a1c": None})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("CDC_A1C", [g["measure"] for g in gaps])

    def test_no_a1c_gap_non_diabetic(self):
        p = _patient(age=55, conditions=[], screenings={"last_a1c": None})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("CDC_A1C", [g["measure"] for g in gaps])

    def test_a1c_gap_poorly_controlled(self):
        recent = (date.today() - timedelta(days=90)).isoformat()
        p = _patient(age=55, conditions=["diabetes"],
                     screenings={"last_a1c": recent, "a1c_value": 10.2})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("CDC_A1C", [g["measure"] for g in gaps])

    # ── Statin (LDL) ────────────────────────────────────────────────
    def test_ldl_gap_cvd_not_on_statin(self):
        p = _patient(age=60, conditions=["cardiovascular_disease"], screenings={"on_statin": False})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("LDL", [g["measure"] for g in gaps])

    def test_no_ldl_gap_on_statin(self):
        p = _patient(age=60, conditions=["cardiovascular_disease"], screenings={"on_statin": True})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("LDL", [g["measure"] for g in gaps])

    def test_no_ldl_gap_no_cvd(self):
        p = _patient(age=60, conditions=[], screenings={"on_statin": False})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("LDL", [g["measure"] for g in gaps])

    # ── Wellness Visit (AWC) ─────────────────────────────────────────
    def test_awc_gap_senior_no_visit(self):
        p = _patient(age=70, screenings={"last_wellness_visit": None})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("AWC", [g["measure"] for g in gaps])

    def test_no_awc_gap_age_64(self):
        p = _patient(age=64, screenings={"last_wellness_visit": None})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("AWC", [g["measure"] for g in gaps])

    # ── Flu (FLU) ────────────────────────────────────────────────────
    def test_flu_gap_over_50_no_shot(self):
        p = _patient(age=55, screenings={"last_flu_shot": None})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("FLU", [g["measure"] for g in gaps])

    def test_no_flu_gap_under_50(self):
        p = _patient(age=45, screenings={"last_flu_shot": None})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("FLU", [g["measure"] for g in gaps])

    # ── Depression (DEP) ─────────────────────────────────────────────
    def test_dep_gap_no_screen(self):
        p = _patient(age=30, screenings={"last_depression_screen": None})
        gaps = self.ident.identify_gaps(p)
        self.assertIn("DEP", [g["measure"] for g in gaps])

    def test_dep_gap_applies_to_all_ages(self):
        for age in (20, 40, 60, 80):
            p = _patient(age=age, screenings={"last_depression_screen": None})
            gaps = self.ident.identify_gaps(p)
            self.assertIn("DEP", [g["measure"] for g in gaps], f"DEP missing for age {age}")

    def test_no_dep_gap_if_recent_screen(self):
        recent = (date.today() - timedelta(days=90)).isoformat()
        p = _patient(age=40, screenings={"last_depression_screen": recent})
        gaps = self.ident.identify_gaps(p)
        self.assertNotIn("DEP", [g["measure"] for g in gaps])


# ============================================================
# RiskStratifier tests
# ============================================================
class TestRiskStratifier(unittest.TestCase):
    def setUp(self):
        self.strat = RiskStratifier()

    def test_result_has_tier_field(self):
        p = _patient(age=60, conditions=["diabetes"])
        gaps = [_gap("CDC_A1C")]
        result = self.strat.stratify(p, gaps)
        self.assertIn("tier", result[0])

    def test_tier_is_valid(self):
        p = _patient(age=60)
        gaps = [_gap("COLO"), _gap("FLU"), _gap("CBP")]
        result = self.strat.stratify(p, gaps)
        valid_tiers = {"CRITICAL", "HIGH", "MODERATE", "ROUTINE"}
        for r in result:
            self.assertIn(r["tier"], valid_tiers)

    def test_priority_score_in_range(self):
        p = _patient(age=65, conditions=["hypertension", "diabetes"],
                     risk_factors={"smoker": True, "bmi_obese": True})
        gaps = [_gap("CDC_A1C"), _gap("CBP"), _gap("COLO")]
        result = self.strat.stratify(p, gaps)
        for r in result:
            self.assertGreaterEqual(r["priority_score"], 0.0)
            self.assertLessEqual(r["priority_score"], 1.0)

    def test_sorted_descending(self):
        p = _patient(age=60, conditions=["hypertension"])
        gaps = [_gap("FLU", score=0.55), _gap("CBP", score=0.95)]
        result = self.strat.stratify(p, gaps)
        scores = [r["priority_score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_gaps_returns_empty(self):
        p = _patient()
        result = self.strat.stratify(p, [])
        self.assertEqual(result, [])

    def test_smoker_boosts_colo_score(self):
        p_s = _patient(age=55, risk_factors={"smoker": True})
        p_n = _patient(age=55, risk_factors={"smoker": False})
        g = [{"measure": "COLO", "reason": "x", "severity": "due"}]
        r_s = self.strat.stratify(p_s, g)[0]["priority_score"]
        r_n = self.strat.stratify(p_n, g)[0]["priority_score"]
        self.assertGreater(r_s, r_n)

    def test_family_hx_boosts_cancer_screening(self):
        p_fhx = _patient(age=55, risk_factors={"family_hx_cancer": True})
        p_none = _patient(age=55, risk_factors={"family_hx_cancer": False})
        g = [{"measure": "BCS", "reason": "x", "severity": "due"}]
        r_fhx = self.strat.stratify(p_fhx, g)[0]["priority_score"]
        r_none = self.strat.stratify(p_none, g)[0]["priority_score"]
        self.assertGreater(r_fhx, r_none)

    def test_overdue_severity_boosts_score(self):
        p = _patient()
        g_due = [{"measure": "COLO", "reason": "x", "severity": "due"}]
        g_over = [{"measure": "COLO", "reason": "x", "severity": "overdue"}]
        s_due = self.strat.stratify(p, g_due)[0]["priority_score"]
        s_over = self.strat.stratify(p, g_over)[0]["priority_score"]
        self.assertGreater(s_over, s_due)

    def test_multiple_conditions_boost(self):
        p_multi = _patient(conditions=["diabetes", "hypertension", "cardiovascular_disease"])
        p_single = _patient(conditions=["diabetes"])
        g = [{"measure": "CDC_A1C", "reason": "x", "severity": "due"}]
        r_m = self.strat.stratify(p_multi, g)[0]["priority_score"]
        r_s = self.strat.stratify(p_single, g)[0]["priority_score"]
        self.assertGreater(r_m, r_s)


# ============================================================
# OutreachDraftingAgent tests
# ============================================================
class TestOutreachDraftingAgent(unittest.TestCase):
    def setUp(self):
        self.drafter = OutreachDraftingAgent()

    def test_generates_package(self):
        p = _patient(age=55, sex="female")
        result = self.drafter.draft_outreach(p, [_gap("BCS")])
        self.assertIsNotNone(result)
        self.assertIn("subject", result)
        self.assertIn("message", result)

    def test_returns_none_for_empty_gaps(self):
        result = self.drafter.draft_outreach(_patient(), [])
        self.assertIsNone(result)

    def test_patient_id_in_result(self):
        p = _patient()
        result = self.drafter.draft_outreach(p, [_gap("FLU")])
        self.assertEqual(result["patient_id"], "TESTPT")

    def test_tier_propagated(self):
        p = _patient()
        g = _gap("CBP", tier="CRITICAL", score=0.97)
        result = self.drafter.draft_outreach(p, [g])
        self.assertEqual(result["tier"], "CRITICAL")

    def test_spanish_patient_gets_spanish_subject(self):
        p = _patient(language="Spanish")
        result = self.drafter.draft_outreach(p, [_gap("BCS")])
        self.assertIsNotNone(result)
        # Spanish subject should differ from English default
        self.assertNotEqual(result["subject"], "Reminder: Annual Mammogram Screening Due")

    def test_english_patient_gets_english_subject(self):
        p = _patient(language="English")
        result = self.drafter.draft_outreach(p, [_gap("BCS")])
        self.assertEqual(result["subject"], "Reminder: Annual Mammogram Screening Due")

    def test_max_two_additional_gaps(self):
        p = _patient()
        gaps = [_gap("CDC_A1C"), _gap("BCS"), _gap("COLO"), _gap("FLU"), _gap("DEP")]
        result = self.drafter.draft_outreach(p, gaps)
        self.assertLessEqual(len(result["additional_gaps"]), 2)

    def test_total_gaps_accurate(self):
        p = _patient()
        gaps = [_gap("CDC_A1C"), _gap("BCS"), _gap("COLO")]
        result = self.drafter.draft_outreach(p, gaps)
        self.assertEqual(result["total_gaps"], 3)

    def test_all_templates_accessible(self):
        """Every HEDIS measure's template should produce a valid outreach package."""
        p = _patient(age=55, sex="female",
                     conditions=["diabetes", "hypertension", "cardiovascular_disease"])
        for measure in HEDIS_MEASURES:
            g = _gap(measure)
            result = self.drafter.draft_outreach(p, [g])
            self.assertIsNotNone(result, f"None result for measure {measure}")
            self.assertIn("subject", result)


# ============================================================
# AnalyticsReporter tests
# ============================================================
class TestAnalyticsReporter(unittest.TestCase):
    def setUp(self):
        self.reporter = AnalyticsReporter()

    def _mock_patients(self, n):
        return [{"patient_id": f"PT{i}"} for i in range(n)]

    def test_summary_keys_present(self):
        pts = self._mock_patients(5)
        gaps = {p["patient_id"]: [] for p in pts}
        report = self.reporter.generate_report(pts, gaps)
        for key in ("total_patients", "patients_with_gaps", "gap_rate_pct",
                    "total_gaps_identified", "avg_gaps_per_patient"):
            self.assertIn(key, report["summary"])

    def test_gap_rate_zero_when_no_gaps(self):
        pts = self._mock_patients(10)
        gaps = {p["patient_id"]: [] for p in pts}
        report = self.reporter.generate_report(pts, gaps)
        self.assertEqual(report["summary"]["gap_rate_pct"], 0.0)

    def test_gap_rate_50_percent(self):
        pts = self._mock_patients(10)
        gaps = {f"PT{i}": [_gap("COLO")] if i < 5 else [] for i in range(10)}
        report = self.reporter.generate_report(pts, gaps)
        self.assertAlmostEqual(report["summary"]["gap_rate_pct"], 50.0)

    def test_critical_count_correct(self):
        pts = self._mock_patients(3)
        gaps = {
            "PT0": [_gap("CBP", tier="CRITICAL")],
            "PT1": [_gap("CDC_A1C", tier="CRITICAL")],
            "PT2": [_gap("FLU", tier="ROUTINE")],
        }
        report = self.reporter.generate_report(pts, gaps)
        self.assertEqual(report["priority_breakdown"]["CRITICAL"], 2)
        self.assertEqual(report["priority_breakdown"]["ROUTINE"], 1)

    def test_vbc_opportunity_positive(self):
        pts = self._mock_patients(20)
        gaps = {f"PT{i}": [_gap("COLO")] for i in range(10)}
        gaps.update({f"PT{i}": [] for i in range(10, 20)})
        report = self.reporter.generate_report(pts, gaps)
        self.assertGreater(report["quality_impact"]["estimated_vbc_opportunity_usd"], 0)

    def test_gap_by_measure_sorted_by_count(self):
        pts = self._mock_patients(5)
        gaps = {
            "PT0": [_gap("COLO"), _gap("FLU")],
            "PT1": [_gap("COLO"), _gap("FLU")],
            "PT2": [_gap("COLO")],
            "PT3": [_gap("FLU")],
            "PT4": [],
        }
        report = self.reporter.generate_report(pts, gaps)
        counts = [g["count"] for g in report["gap_by_measure"]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_empty_population(self):
        report = self.reporter.generate_report([], {})
        self.assertEqual(report["summary"]["total_patients"], 0)
        self.assertEqual(report["summary"]["gap_rate_pct"], 0.0)


# ============================================================
# End-to-end Orchestrator tests
# ============================================================
class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.agent = HEDISCareGapOrchestrator(verbose=False)

    def test_full_run_returns_expected_keys(self):
        result = self.agent.run(patient_count=20, top_n=5)
        for key in ("patients", "patients_with_gaps", "total_gaps",
                    "outreach_packages", "analytics", "stratified_gaps"):
            self.assertIn(key, result)

    def test_patient_count_correct(self):
        result = self.agent.run(patient_count=12, top_n=5)
        self.assertEqual(result["patients"], 12)

    def test_outreach_within_top_n(self):
        result = self.agent.run(patient_count=30, top_n=8)
        self.assertLessEqual(len(result["outreach_packages"]), 8)

    def test_analytics_summary_totals_consistent(self):
        result = self.agent.run(patient_count=25, top_n=5)
        s = result["analytics"]["summary"]
        self.assertEqual(s["total_patients"], 25)
        self.assertLessEqual(s["patients_with_gaps"], 25)
        self.assertGreaterEqual(s["total_gaps_identified"], 0)

    def test_outreach_packages_have_required_fields(self):
        result = self.agent.run(patient_count=20, top_n=5)
        for pkg in result["outreach_packages"]:
            for field in ("patient_id", "tier", "subject", "message", "total_gaps"):
                self.assertIn(field, pkg, f"Missing field {field} in outreach package")

    def test_gap_rate_between_0_and_100(self):
        result = self.agent.run(patient_count=40, top_n=10)
        rate = result["analytics"]["summary"]["gap_rate_pct"]
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 100.0)

    def test_all_outreach_tiers_valid(self):
        result = self.agent.run(patient_count=50, top_n=15)
        valid = {"CRITICAL", "HIGH", "MODERATE", "ROUTINE"}
        for pkg in result["outreach_packages"]:
            self.assertIn(pkg["tier"], valid)


# ============================================================
# ElephantAgent tests
# ============================================================
class TestElephantAgent(unittest.TestCase):
    def setUp(self):
        self.elephant = ElephantAgent()
        self.loader   = PatientProfileLoader()

    def _sample_patients(self, n=10):
        return self.loader.load_synthetic_panel(n)

    def test_build_context_keys(self):
        pts = self._sample_patients(10)
        ctx = self.elephant.build_context(pts, HEDIS_MEASURES)
        for k in ("patient_count", "measure_count", "measures",
                  "condition_distribution", "age_distribution"):
            self.assertIn(k, ctx)

    def test_build_context_patient_count(self):
        pts = self._sample_patients(7)
        ctx = self.elephant.build_context(pts, HEDIS_MEASURES)
        self.assertEqual(ctx["patient_count"], 7)

    def test_build_context_measure_count(self):
        ctx = self.elephant.build_context([], HEDIS_MEASURES)
        self.assertEqual(ctx["measure_count"], len(HEDIS_MEASURES))

    def test_build_context_age_distribution(self):
        pts = self._sample_patients(20)
        ctx = self.elephant.build_context(pts, HEDIS_MEASURES)
        dist = ctx["age_distribution"]
        self.assertIn("18-44", dist)
        self.assertIn("45-64", dist)
        self.assertIn("65+",   dist)
        total = sum(dist.values())
        self.assertEqual(total, 20)

    def test_resolve_hedis_critiques_removes_violation(self):
        gap_report = {
            "stratified_gaps": {
                "PT001": [
                    {"measure": "BCS", "tier": "HIGH", "priority_score": 0.8},
                    {"measure": "COLO", "tier": "HIGH", "priority_score": 0.7},
                ]
            }
        }
        violations = [{"patient_id": "PT001", "measure": "BCS",
                       "violation": "Age 30 outside BCS range 50-74"}]
        result = self.elephant.resolve_hedis_critiques(gap_report, violations)
        measures_left = [g["measure"] for g in result["stratified_gaps"]["PT001"]]
        self.assertNotIn("BCS", measures_left)
        self.assertIn("COLO", measures_left)

    def test_resolve_hedis_critiques_tracks_fix_count(self):
        gap_report = {
            "stratified_gaps": {
                "PT001": [{"measure": "BCS", "tier": "HIGH", "priority_score": 0.8}]
            }
        }
        violations = [{"patient_id": "PT001", "measure": "BCS", "violation": "x"}]
        result = self.elephant.resolve_hedis_critiques(gap_report, violations)
        self.assertEqual(result["_hedis_fixes"], 1)

    def test_resolve_hedis_critiques_unknown_patient_ignored(self):
        gap_report = {"stratified_gaps": {"PT001": []}}
        violations = [{"patient_id": "UNKNOWN", "measure": "BCS", "violation": "x"}]
        result = self.elephant.resolve_hedis_critiques(gap_report, violations)
        self.assertEqual(result["_hedis_fixes"], 0)

    def test_resolve_priority_critiques_downgrades_critical(self):
        gap_report = {
            "stratified_gaps": {
                "PT001": [{"measure": "FLU", "tier": "CRITICAL", "priority_score": 0.95}]
            }
        }
        unjustified = [{"patient_id": "PT001", "measure": "FLU", "issue": "inflated"}]
        result = self.elephant.resolve_priority_critiques(gap_report, unjustified)
        gap = result["stratified_gaps"]["PT001"][0]
        self.assertEqual(gap["tier"], "HIGH")
        self.assertLessEqual(gap["priority_score"], 0.89)

    def test_resolve_priority_critiques_tracks_fix_count(self):
        gap_report = {
            "stratified_gaps": {
                "PT001": [{"measure": "DEP", "tier": "CRITICAL", "priority_score": 0.91}]
            }
        }
        unjustified = [{"patient_id": "PT001", "measure": "DEP", "issue": "x"}]
        result = self.elephant.resolve_priority_critiques(gap_report, unjustified)
        self.assertEqual(result["_priority_fixes"], 1)

    def test_resolve_priority_critiques_leaves_high_untouched(self):
        gap_report = {
            "stratified_gaps": {
                "PT001": [{"measure": "FLU", "tier": "HIGH", "priority_score": 0.78}]
            }
        }
        unjustified = [{"patient_id": "PT001", "measure": "FLU", "issue": "x"}]
        result = self.elephant.resolve_priority_critiques(gap_report, unjustified)
        # HIGH tier should remain unchanged (only CRITICAL is downgraded)
        gap = result["stratified_gaps"]["PT001"][0]
        self.assertEqual(gap["tier"], "HIGH")
        self.assertEqual(gap["priority_score"], 0.78)


# ============================================================
# HEDISComplianceGoldfish tests
# ============================================================
class TestHEDISComplianceGoldfish(unittest.TestCase):
    def setUp(self):
        self.goldfish = HEDISComplianceGoldfish()

    def _stratified(self, pid, gaps_list):
        return {pid: gaps_list}

    def _pts_by_id(self, **kwargs):
        p = _patient(**kwargs)
        p["patient_id"] = "PT001"
        return {"PT001": p}

    def test_pass_on_empty_gaps(self):
        result = self.goldfish.review({}, {})
        self.assertTrue(result["passed"])
        self.assertEqual(result["violation_count"], 0)

    def test_age_below_min_flagged(self):
        # BCS: age_min=50; age=40 → violation
        stratified = self._stratified("PT001", [
            {"measure": "BCS", "tier": "HIGH", "priority_score": 0.8}
        ])
        pts = self._pts_by_id(age=40, sex="female")
        result = self.goldfish.review(stratified, pts)
        self.assertFalse(result["passed"])
        self.assertGreater(result["violation_count"], 0)

    def test_age_above_max_flagged(self):
        # BCS: age_max=74; age=80 → violation
        stratified = self._stratified("PT001", [
            {"measure": "BCS", "tier": "HIGH", "priority_score": 0.8}
        ])
        pts = self._pts_by_id(age=80, sex="female")
        result = self.goldfish.review(stratified, pts)
        self.assertFalse(result["passed"])

    def test_wrong_sex_flagged(self):
        # BCS requires female; male patient → violation
        stratified = self._stratified("PT001", [
            {"measure": "BCS", "tier": "HIGH", "priority_score": 0.8}
        ])
        pts = self._pts_by_id(age=55, sex="male")
        result = self.goldfish.review(stratified, pts)
        self.assertFalse(result["passed"])
        violations = [v["violation"] for v in result["violations"]]
        self.assertTrue(any("ineligible" in v.lower() for v in violations))

    def test_missing_condition_flagged(self):
        # CBP requires hypertension; patient has none → violation
        stratified = self._stratified("PT001", [
            {"measure": "CBP", "tier": "HIGH", "priority_score": 0.8}
        ])
        pts = self._pts_by_id(age=55, sex="male", conditions=[])
        result = self.goldfish.review(stratified, pts)
        self.assertFalse(result["passed"])
        violations = [v["violation"] for v in result["violations"]]
        self.assertTrue(any("hypertension" in v.lower() for v in violations))

    def test_valid_patient_passes(self):
        # COLO: age=55, sex=any → valid
        stratified = self._stratified("PT001", [
            {"measure": "COLO", "tier": "HIGH", "priority_score": 0.8}
        ])
        pts = self._pts_by_id(age=55, sex="male")
        result = self.goldfish.review(stratified, pts)
        self.assertTrue(result["passed"])

    def test_result_has_required_keys(self):
        result = self.goldfish.review({}, {})
        for k in ("critic", "passed", "violations", "violation_count", "verdict"):
            self.assertIn(k, result)

    def test_critic_name_correct(self):
        result = self.goldfish.review({}, {})
        self.assertEqual(result["critic"], "HEDISComplianceGoldfish")


# ============================================================
# ClinicalPriorityGoldfish tests
# ============================================================
class TestClinicalPriorityGoldfish(unittest.TestCase):
    def setUp(self):
        self.goldfish = ClinicalPriorityGoldfish()

    def _run(self, measure, tier, age, conditions=None, risk_factors=None):
        pid = "PT001"
        stratified = {pid: [{"measure": measure, "tier": tier,
                              "priority_score": 0.95, "reason": "test"}]}
        pts_by_id = {pid: _patient(age=age, conditions=conditions or [],
                                   risk_factors=risk_factors or {})}
        pts_by_id[pid]["patient_id"] = pid
        return self.goldfish.review(stratified, pts_by_id)

    def test_pass_on_no_critical_flags(self):
        # HIGH tier for FLU → no complaint
        result = self._run("FLU", "HIGH", age=40)
        self.assertTrue(result["passed"])

    def test_pass_critical_for_high_base_priority_measure(self):
        # CBP is NOT in LOW_BASE_PRIORITY → CRITICAL always OK
        result = self._run("CBP", "CRITICAL", age=40)
        self.assertTrue(result["passed"])

    def test_fail_unjustified_critical_low_priority_young(self):
        # FLU + young + no risk → unjustified CRITICAL
        result = self._run("FLU", "CRITICAL", age=35, conditions=[],
                           risk_factors={"smoker": False, "family_hx_cancer": False, "bmi_obese": False})
        self.assertFalse(result["passed"])
        self.assertGreater(result["unjustified_count"], 0)

    def test_pass_critical_low_priority_if_age_65(self):
        # FLU + age 65+ → justified CRITICAL
        result = self._run("FLU", "CRITICAL", age=65)
        self.assertTrue(result["passed"])

    def test_pass_critical_low_priority_if_smoker(self):
        # DEP + smoker → justified CRITICAL
        result = self._run("DEP", "CRITICAL", age=40, risk_factors={"smoker": True})
        self.assertTrue(result["passed"])

    def test_pass_critical_low_priority_if_2_conditions(self):
        # AWC + 2 conditions → justified CRITICAL
        result = self._run("AWC", "CRITICAL", age=55,
                           conditions=["hypertension", "diabetes"])
        self.assertTrue(result["passed"])

    def test_pass_critical_low_priority_if_family_hx(self):
        # CCS + family_hx_cancer → justified CRITICAL
        result = self._run("CCS", "CRITICAL", age=45,
                           risk_factors={"family_hx_cancer": True})
        self.assertTrue(result["passed"])

    def test_result_has_required_keys(self):
        result = self._run("FLU", "HIGH", age=40)
        for k in ("critic", "passed", "unjustified", "unjustified_count", "verdict"):
            self.assertIn(k, result)

    def test_critic_name_correct(self):
        result = self._run("FLU", "HIGH", age=40)
        self.assertEqual(result["critic"], "ClinicalPriorityGoldfish")


# ============================================================
# OutreachClarityGoldfish tests
# ============================================================
class TestOutreachClarityGoldfish(unittest.TestCase):
    def setUp(self):
        self.goldfish = OutreachClarityGoldfish()

    def _pkg(self, subject="Reminder: Get Your Screening", message=None,
             closing="Call our office at 555-0100", primary_gap_code="COLO",
             patient_id="PT001"):
        return {
            "patient_id": patient_id,
            "subject": subject,
            "message": message if message is not None else (
                "Dear Patient, our records show you are due for an important preventive "
                "health screening. Please schedule your appointment at your earliest convenience."
            ),
            "closing": closing,
            "primary_gap_code": primary_gap_code,
        }

    def test_pass_clean_package(self):
        result = self.goldfish.review([self._pkg()])
        self.assertTrue(result["passed"])
        self.assertEqual(result["issue_count"], 0)

    def test_fail_empty_subject(self):
        result = self.goldfish.review([self._pkg(subject="")])
        self.assertFalse(result["passed"])
        issues = [i["issue"] for i in result["issues"]]
        self.assertTrue(any("subject" in i.lower() for i in issues))

    def test_fail_whitespace_only_subject(self):
        result = self.goldfish.review([self._pkg(subject="   ")])
        self.assertFalse(result["passed"])

    def test_fail_message_too_short(self):
        result = self.goldfish.review([self._pkg(message="Short.")])
        self.assertFalse(result["passed"])
        issues = [i["issue"] for i in result["issues"]]
        self.assertTrue(any("short" in i.lower() for i in issues))

    def test_fail_missing_closing(self):
        result = self.goldfish.review([self._pkg(closing="")])
        self.assertFalse(result["passed"])
        issues = [i["issue"] for i in result["issues"]]
        self.assertTrue(any("closing" in i.lower() or "call-to-action" in i.lower()
                            for i in issues))

    def test_fail_missing_gap_code(self):
        result = self.goldfish.review([self._pkg(primary_gap_code=None)])
        self.assertFalse(result["passed"])
        issues = [i["issue"] for i in result["issues"]]
        self.assertTrue(any("gap code" in i.lower() for i in issues))

    def test_empty_packages_list_passes(self):
        result = self.goldfish.review([])
        self.assertTrue(result["passed"])
        self.assertEqual(result["issue_count"], 0)

    def test_multiple_packages_all_valid(self):
        pkgs = [self._pkg(patient_id=f"PT{i:03d}") for i in range(5)]
        result = self.goldfish.review(pkgs)
        self.assertTrue(result["passed"])

    def test_result_has_required_keys(self):
        result = self.goldfish.review([])
        for k in ("critic", "passed", "issues", "issue_count", "verdict"):
            self.assertIn(k, result)

    def test_critic_name_correct(self):
        result = self.goldfish.review([])
        self.assertEqual(result["critic"], "OutreachClarityGoldfish")


# ============================================================
# EGMHEDISOrchestrator tests
# ============================================================
class TestEGMOrchestrator(unittest.TestCase):
    def setUp(self):
        self.agent = EGMHEDISOrchestrator(verbose=False, max_rounds=2)

    def test_full_run_returns_expected_keys(self):
        result = self.agent.run(patient_count=20, top_n=5)
        for key in ("patients", "patients_with_gaps", "total_gaps",
                    "outreach_packages", "analytics", "stratified_gaps",
                    "egm_review_log", "final_version", "egm_total_fixed"):
            self.assertIn(key, result)

    def test_final_version_is_string(self):
        result = self.agent.run(patient_count=15, top_n=5)
        self.assertIsInstance(result["final_version"], str)
        # version should be like "1.0" or "2.0"
        self.assertIn(".", result["final_version"])

    def test_review_log_is_list(self):
        result = self.agent.run(patient_count=15, top_n=5)
        self.assertIsInstance(result["egm_review_log"], list)
        self.assertGreater(len(result["egm_review_log"]), 0)

    def test_review_log_rounds_within_max(self):
        agent = EGMHEDISOrchestrator(verbose=False, max_rounds=2)
        result = agent.run(patient_count=15, top_n=5)
        self.assertLessEqual(len(result["egm_review_log"]), 2)

    def test_review_log_entry_keys(self):
        result = self.agent.run(patient_count=15, top_n=5)
        for entry in result["egm_review_log"]:
            for k in ("round", "version", "hedis_compliance",
                      "clinical_priority", "outreach_clarity", "all_passed"):
                self.assertIn(k, entry)

    def test_egm_total_fixed_non_negative(self):
        result = self.agent.run(patient_count=15, top_n=5)
        self.assertGreaterEqual(result["egm_total_fixed"], 0)

    def test_outreach_within_top_n(self):
        result = self.agent.run(patient_count=20, top_n=6)
        self.assertLessEqual(len(result["outreach_packages"]), 6)

    def test_analytics_summary_consistent(self):
        result = self.agent.run(patient_count=20, top_n=5)
        s = result["analytics"]["summary"]
        self.assertEqual(s["total_patients"], 20)
        self.assertLessEqual(s["patients_with_gaps"], 20)

    def test_goldfish_critics_named_correctly_in_log(self):
        result = self.agent.run(patient_count=10, top_n=3)
        entry = result["egm_review_log"][0]
        self.assertEqual(entry["hedis_compliance"]["critic"], "HEDISComplianceGoldfish")
        self.assertEqual(entry["clinical_priority"]["critic"], "ClinicalPriorityGoldfish")
        self.assertEqual(entry["outreach_clarity"]["critic"], "OutreachClarityGoldfish")

    def test_max_rounds_1_runs_single_round(self):
        agent = EGMHEDISOrchestrator(verbose=False, max_rounds=1)
        result = agent.run(patient_count=10, top_n=3)
        self.assertEqual(len(result["egm_review_log"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
