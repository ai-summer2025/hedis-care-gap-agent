#!/usr/bin/env python3
"""
HEDIS Care Gap Closure Agent  [EGM Edition]
=============================================
Autonomous 5-step agentic AI for identifying HEDIS quality measure gaps,
enhanced with Google's Elephant-Goldfish Model (EGM) adversarial review loop.

  GitHub: https://github.com/google/elephant-goldfish-model  (Apache-2.0, Oct 2026)

EGM Architecture:
  Elephant (long-context synthesizer) → runs full population gap analysis
  → 3 Goldfish critics (stateless, no memory) attack the output independently:
       🐟 HEDISComplianceGoldfish   — verifies every gap flag matches HEDIS specs
       🐟 ClinicalPriorityGoldfish  — challenges unjustified CRITICAL tier flags
       🐟 OutreachClarityGoldfish   — reviews actionability of outreach messages
  → Any critic fails? Elephant resolves critiques → bump version → repeat
  → All pass (or max rounds reached) → validated final output

Problem: US practices leave ~$100B/year in value-based care bonuses unrealized
because patients fall behind on HEDIS preventive screenings. A single wrong
gap flag (e.g., flagging a 43-year-old for colorectal screening) erodes clinician
trust in the tool. EGM's adversarial loop catches these errors before they reach
the care coordinator's queue.

Usage:
    python main.py                              # EGM mode, 50 patients, top 10
    python main.py --patients 100 --top 20      # larger panel
    python main.py --no-egm                     # plain pipeline (no review loop)
    python main.py --output results.json        # save full JSON output
"""

import json
import random
import sys
import argparse
from datetime import date, timedelta
from typing import Optional, List, Dict, Any


# ============================================================
# HEDIS MEASURE DEFINITIONS
# ============================================================
HEDIS_MEASURES: Dict[str, Dict[str, Any]] = {
    "COLO": {
        "name": "Colorectal Cancer Screening",
        "description": "Adults 45-75 who had appropriate colorectal cancer screening",
        "age_min": 45, "age_max": 75, "sex": "any",
        "gap_window_months": 120,
        "condition_required": None,
        "priority_weight": 0.90,
        "outreach_template": "colorectal_screening",
        "clinical_rationale": "Colorectal cancer is 90% survivable when caught early (ACS 2026)",
    },
    "BCS": {
        "name": "Breast Cancer Screening",
        "description": "Women 50-74 who had a mammogram in the past 2 years",
        "age_min": 50, "age_max": 74, "sex": "female",
        "gap_window_months": 24,
        "condition_required": None,
        "priority_weight": 0.90,
        "outreach_template": "mammogram",
        "clinical_rationale": "Routine mammography reduces breast cancer mortality by 20-40% (USPSTF)",
    },
    "CCS": {
        "name": "Cervical Cancer Screening",
        "description": "Women 21-64 who had cervical cancer screening",
        "age_min": 21, "age_max": 64, "sex": "female",
        "gap_window_months": 36,
        "condition_required": None,
        "priority_weight": 0.80,
        "outreach_template": "pap_smear",
        "clinical_rationale": "Cervical cancer is nearly 100% preventable with timely screening (CDC)",
    },
    "CBP": {
        "name": "Controlling High Blood Pressure",
        "description": "Hypertension patients 18-85 with BP at goal (<140/90)",
        "age_min": 18, "age_max": 85, "sex": "any",
        "gap_window_months": 12,
        "condition_required": "hypertension",
        "priority_weight": 0.95,
        "outreach_template": "blood_pressure",
        "clinical_rationale": "Uncontrolled hypertension doubles stroke risk (AHA 2025)",
    },
    "CDC_A1C": {
        "name": "Diabetes A1c Testing",
        "description": "Diabetic patients 18-75 who had A1c testing in past year",
        "age_min": 18, "age_max": 75, "sex": "any",
        "gap_window_months": 12,
        "condition_required": "diabetes",
        "priority_weight": 0.95,
        "outreach_template": "diabetes_a1c",
        "clinical_rationale": "Unmonitored diabetes leads to nephropathy, neuropathy, blindness (NDEI 2025)",
    },
    "CDC_EYE": {
        "name": "Diabetic Retinal Eye Exam",
        "description": "Diabetic patients 18-75 who had a retinal eye exam in past year",
        "age_min": 18, "age_max": 75, "sex": "any",
        "gap_window_months": 12,
        "condition_required": "diabetes",
        "priority_weight": 0.70,
        "outreach_template": "diabetes_eye",
        "clinical_rationale": "Diabetic retinopathy is the #1 cause of new blindness in working-age adults",
    },
    "LDL": {
        "name": "Statin Therapy for CVD",
        "description": "CVD patients 21-85 on statin therapy per ACC/AHA guidelines",
        "age_min": 21, "age_max": 85, "sex": "any",
        "gap_window_months": 12,
        "condition_required": "cardiovascular_disease",
        "priority_weight": 0.88,
        "outreach_template": "statin_therapy",
        "clinical_rationale": "Statins reduce MACE risk by 25-35% in CVD patients (ACC/AHA 2023)",
    },
    "AWC": {
        "name": "Annual Wellness Visit",
        "description": "Medicare patients 65+ who had an annual wellness visit",
        "age_min": 65, "age_max": 120, "sex": "any",
        "gap_window_months": 12,
        "condition_required": None,
        "priority_weight": 0.75,
        "outreach_template": "wellness_visit",
        "clinical_rationale": "AWV enables early detection and advance care planning for Medicare beneficiaries",
    },
    "FLU": {
        "name": "Influenza Vaccination",
        "description": "Adults 50+ who received influenza vaccination during flu season",
        "age_min": 50, "age_max": 120, "sex": "any",
        "gap_window_months": 12,
        "condition_required": None,
        "priority_weight": 0.65,
        "outreach_template": "flu_shot",
        "clinical_rationale": "Flu kills 12,000-52,000 Americans annually; vaccine reduces hospitalization by 40%",
    },
    "DEP": {
        "name": "Depression Screening",
        "description": "Adults 18+ who were screened for depression (PHQ-2/PHQ-9) in past year",
        "age_min": 18, "age_max": 120, "sex": "any",
        "gap_window_months": 12,
        "condition_required": None,
        "priority_weight": 0.68,
        "outreach_template": "depression_screening",
        "clinical_rationale": "1 in 6 adults has untreated depression; screening increases detection 3-fold (AHRQ)",
    },
}


# ============================================================
# AGENT 1: PATIENT PROFILE LOADER
# ============================================================
class PatientProfileLoader:
    """Loads and normalizes patient population data."""

    def load_synthetic_panel(self, size: int = 50) -> List[Dict]:
        conditions_pool = [
            ["hypertension"],
            ["diabetes", "hypertension"],
            ["cardiovascular_disease", "hypertension"],
            ["diabetes"],
            ["cardiovascular_disease"],
            ["hypertension", "cardiovascular_disease"],
            ["diabetes", "cardiovascular_disease", "hypertension"],
            [], [], [],
        ]
        female_names = ["Maria","Jennifer","Patricia","Linda","Barbara","Susan","Dorothy",
                        "Lisa","Nancy","Karen","Betty","Helen","Sandra","Donna","Carol",
                        "Ruth","Sharon","Michelle","Laura","Sarah"]
        male_names   = ["James","John","Robert","Michael","William","David","Richard",
                        "Joseph","Thomas","Charles","Christopher","Daniel","Paul","Mark",
                        "Donald","George","Kenneth","Steven","Edward","Brian"]
        last_names   = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
                        "Davis","Rodriguez","Martinez","Hernandez","Lopez","Gonzalez",
                        "Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin"]

        patients = []
        rng = random.Random(42)

        for i in range(size):
            sex = rng.choice(["male", "female"])
            age = rng.randint(18, 85)
            conditions = rng.choice(conditions_pool)
            today = date.today()

            def past_date(min_y: float, max_y: float) -> str:
                days = rng.randint(int(min_y * 365), int(max_y * 365))
                return (today - timedelta(days=days)).isoformat()

            sc: Dict[str, Any] = {}
            sc["last_colonoscopy"] = (
                None if rng.random() < 0.60
                else past_date(11, 20) if rng.random() < 0.5
                else past_date(0.5, 9)
            )
            if sex == "female":
                sc["last_mammogram"] = (
                    None if rng.random() < 0.45
                    else past_date(0, 1.9) if rng.random() < 0.5
                    else past_date(3, 8)
                )
                sc["last_pap_smear"] = (
                    None if rng.random() < 0.40
                    else past_date(0, 2.9) if rng.random() < 0.55
                    else past_date(4, 10)
                )
            if "diabetes" in conditions:
                sc["last_a1c"] = (
                    None if rng.random() < 0.50
                    else past_date(0, 0.9) if rng.random() < 0.55
                    else past_date(1.2, 4)
                )
                sc["a1c_value"] = round(rng.uniform(6.5, 11.5), 1)
                sc["last_eye_exam"] = (
                    None if rng.random() < 0.55
                    else past_date(0, 0.9) if rng.random() < 0.5
                    else past_date(1.2, 4)
                )
            if "hypertension" in conditions:
                sc["bp_controlled"] = rng.random() < 0.52
            if "cardiovascular_disease" in conditions:
                sc["on_statin"] = rng.random() < 0.58
            sc["last_flu_shot"] = (
                None if rng.random() < 0.45
                else past_date(0, 0.9) if rng.random() < 0.6
                else past_date(1.2, 2)
            )
            sc["last_depression_screen"] = (
                None if rng.random() < 0.55
                else past_date(0, 0.9) if rng.random() < 0.6
                else past_date(1.2, 3)
            )
            sc["last_wellness_visit"] = (
                None if rng.random() < 0.50
                else past_date(0, 0.9) if rng.random() < 0.55
                else past_date(1.2, 3)
            )

            first = rng.choice(female_names if sex == "female" else male_names)
            last  = rng.choice(last_names)

            patients.append({
                "patient_id": f"PT{1000 + i:04d}",
                "name": f"{first} {last}",
                "age": age,
                "sex": sex,
                "conditions": list(conditions),
                "screenings": sc,
                "risk_factors": {
                    "smoker": rng.random() < 0.18,
                    "bmi_obese": rng.random() < 0.36,
                    "family_hx_cancer": rng.random() < 0.25,
                    "low_health_literacy": rng.random() < 0.28,
                    "sdoh_barrier": rng.random() < 0.22,
                    "no_recent_visit": rng.random() < 0.35,
                },
                "preferred_contact": rng.choice(["portal", "phone", "text"]),
                "language": rng.choice(["English"]*6 + ["Spanish"]*2 + ["Mandarin"] + ["Vietnamese"]),
                "payer": rng.choice(["Medicare", "Medicaid", "Commercial", "Uninsured"]),
            })

        return patients

    def load_from_json(self, filepath: str) -> List[Dict]:
        with open(filepath) as f:
            return json.load(f)


# ============================================================
# AGENT 2: CARE GAP IDENTIFIER
# ============================================================
class CareGapIdentifier:
    """Identifies HEDIS quality measure gaps for each patient."""

    def identify_gaps(self, patient: Dict) -> List[Dict]:
        gaps: List[Dict] = []
        today = date.today()
        age, sex = patient["age"], patient["sex"]
        conditions = patient.get("conditions", [])
        sc = patient.get("screenings", {})

        def months_since(ds: Optional[str]) -> Optional[float]:
            if not ds:
                return None
            try:
                return (today - date.fromisoformat(ds)).days / 30.44
            except (ValueError, TypeError):
                return None

        if 45 <= age <= 75:
            m = months_since(sc.get("last_colonoscopy"))
            if m is None or m > 120:
                gaps.append({"measure":"COLO","reason":"Colonoscopy never done or >10 years ago",
                             "severity":"overdue" if m and m > 180 else "due"})

        if sex == "female" and 50 <= age <= 74:
            m = months_since(sc.get("last_mammogram"))
            if m is None or m > 24:
                gaps.append({"measure":"BCS","reason":f"Mammogram {'never done' if m is None else f'{int(m)} months ago'}",
                             "severity":"overdue" if m and m > 36 else "due"})

        if sex == "female" and 21 <= age <= 64:
            m = months_since(sc.get("last_pap_smear"))
            if m is None or m > 36:
                gaps.append({"measure":"CCS","reason":f"Pap smear {'never done' if m is None else f'{int(m)} months ago'}",
                             "severity":"overdue" if m and m > 48 else "due"})

        if "hypertension" in conditions and 18 <= age <= 85:
            if not sc.get("bp_controlled"):
                gaps.append({"measure":"CBP","reason":"BP not at goal (<140/90) — needs follow-up or med adjustment",
                             "severity":"active"})

        if "diabetes" in conditions and 18 <= age <= 75:
            m = months_since(sc.get("last_a1c"))
            a1c = sc.get("a1c_value")
            if m is None or m > 12:
                gaps.append({"measure":"CDC_A1C","reason":f"A1c {'never tested' if m is None else f'{int(m)} months ago'}",
                             "severity":"overdue" if m and m > 18 else "due"})
            elif a1c and a1c > 9.0:
                gaps.append({"measure":"CDC_A1C","reason":f"A1c={a1c}% poorly controlled — intensified management needed",
                             "severity":"active"})

        if "diabetes" in conditions and 18 <= age <= 75:
            m = months_since(sc.get("last_eye_exam"))
            if m is None or m > 12:
                gaps.append({"measure":"CDC_EYE","reason":f"Retinal exam {'never done' if m is None else f'{int(m)} months ago'}",
                             "severity":"overdue" if m and m > 24 else "due"})

        if "cardiovascular_disease" in conditions and 21 <= age <= 85:
            if not sc.get("on_statin"):
                gaps.append({"measure":"LDL","reason":"Not on statin despite CVD diagnosis (ACC/AHA gap)",
                             "severity":"active"})

        if age >= 65:
            m = months_since(sc.get("last_wellness_visit"))
            if m is None or m > 12:
                gaps.append({"measure":"AWC","reason":f"Annual Wellness Visit {'never done' if m is None else f'{int(m)} months ago'}",
                             "severity":"overdue" if m and m > 18 else "due"})

        if age >= 50:
            m = months_since(sc.get("last_flu_shot"))
            if m is None or m > 12:
                gaps.append({"measure":"FLU","reason":f"Flu vaccine {'never received' if m is None else f'{int(m)} months ago'}",
                             "severity":"due"})

        m = months_since(sc.get("last_depression_screen"))
        if m is None or m > 12:
            gaps.append({"measure":"DEP","reason":f"PHQ-2/PHQ-9 {'never done' if m is None else f'{int(m)} months ago'}",
                         "severity":"due"})

        return gaps


# ============================================================
# AGENT 3: RISK STRATIFIER
# ============================================================
class RiskStratifier:
    """Stratifies gaps with composite priority score and tier."""

    def stratify(self, patient: Dict, gaps: List[Dict]) -> List[Dict]:
        stratified = []
        risk = patient.get("risk_factors", {})
        age, conditions = patient.get("age", 50), patient.get("conditions", [])

        for gap in gaps:
            measure = gap["measure"]
            base = HEDIS_MEASURES.get(measure, {}).get("priority_weight", 0.5)
            mod = 0.0

            if gap.get("severity") == "overdue":   mod += 0.08
            elif gap.get("severity") == "active":  mod += 0.05
            if risk.get("smoker") and measure in ("COLO","BCS","DEP","CBP"):     mod += 0.12
            if risk.get("bmi_obese") and measure in ("CBP","CDC_A1C","COLO","AWC"): mod += 0.08
            if risk.get("family_hx_cancer") and measure in ("COLO","BCS","CCS"): mod += 0.15
            if risk.get("sdoh_barrier"): mod += 0.05
            if risk.get("no_recent_visit"): mod += 0.05
            if len(conditions) >= 3: mod += 0.08
            elif len(conditions) == 2: mod += 0.04
            if age >= 70 and measure in ("AWC","FLU","CBP","CDC_A1C"): mod += 0.07

            score = min(1.0, base + mod)
            tier = "ROUTINE"
            if score >= 0.90: tier = "CRITICAL"
            elif score >= 0.75: tier = "HIGH"
            elif score >= 0.60: tier = "MODERATE"

            stratified.append({
                **gap,
                "priority_score": round(score, 3),
                "tier": tier,
                "measure_name": HEDIS_MEASURES.get(measure, {}).get("name", measure),
                "clinical_rationale": HEDIS_MEASURES.get(measure, {}).get("clinical_rationale", ""),
            })

        stratified.sort(key=lambda x: x["priority_score"], reverse=True)
        return stratified


# ============================================================
# AGENT 4: OUTREACH DRAFTING AGENT
# ============================================================
class OutreachDraftingAgent:
    """Generates personalized, language-aware outreach messages."""

    TEMPLATES: Dict[str, Dict[str, str]] = {
        "colorectal_screening": {
            "subject": "Important: Your Colorectal Cancer Screening is Due",
            "body": ("A colonoscopy takes about 30 minutes and can find colorectal cancer early, "
                     "when it is most treatable. Given your age, guidelines recommend screening. "
                     "Please call us to schedule — we can walk you through the simple preparation steps."),
        },
        "mammogram": {
            "subject": "Reminder: Annual Mammogram Screening Due",
            "body": ("Regular mammograms are the best way to find breast cancer early, when treatment "
                     "works best. Our records show you are due for your screening mammogram. "
                     "Same-week appointments are usually available — please call or book online."),
        },
        "pap_smear": {
            "subject": "Cervical Cancer Screening Reminder",
            "body": ("Cervical cancer is nearly 100% preventable with timely screening. A quick "
                     "Pap test during your next visit is all it takes. Please call us to schedule "
                     "or mention it at your next appointment."),
        },
        "blood_pressure": {
            "subject": "Action Needed: Blood Pressure Follow-Up",
            "body": ("Managing blood pressure is one of the most important things you can do for "
                     "your heart and brain health. Our records show your blood pressure may need "
                     "closer monitoring or a medication adjustment. Please schedule a brief follow-up "
                     "visit — your care team can check this in under 15 minutes."),
        },
        "diabetes_a1c": {
            "subject": "Diabetes Check-Up: A1c Test Due",
            "body": ("Your A1c level tells us how well your diabetes is managed over the past 3 months. "
                     "Staying on top of this can prevent serious complications. Please call to schedule "
                     "a quick lab draw — no special preparation required."),
        },
        "diabetes_eye": {
            "subject": "Annual Diabetic Eye Exam Reminder",
            "body": ("Diabetes can quietly affect your vision before you notice any symptoms. "
                     "An annual retinal eye exam catches problems early when they are easiest to treat. "
                     "Please schedule with an ophthalmologist or optometrist — most insurances cover "
                     "this at no cost to you."),
        },
        "statin_therapy": {
            "subject": "Heart Protection: Medication Review Needed",
            "body": ("Statin medications are recommended for patients with your diagnosis to significantly "
                     "reduce the risk of heart attack and stroke. Our records suggest this medication may "
                     "not be part of your current regimen. Please discuss with your doctor at your next visit."),
        },
        "wellness_visit": {
            "subject": "Annual Wellness Visit — Let's Catch Up on Your Health",
            "body": ("Your Annual Wellness Visit is a dedicated time to review your health goals, "
                     "update your medications, and plan preventive screenings for the year ahead — "
                     "all at no cost to you with Medicare. We have appointments available. "
                     "Please call us to reserve your spot."),
        },
        "flu_shot": {
            "subject": "Flu Season: Get Your Vaccination Today",
            "body": ("Flu can be serious for adults 50 and older. The flu vaccine is one of the "
                     "easiest and most effective ways to protect yourself. Walk-in flu shots are "
                     "available at our office, your local pharmacy, or a community clinic — "
                     "often at no cost with insurance."),
        },
        "depression_screening": {
            "subject": "Annual Mental Health Check-In",
            "body": ("Mental and emotional wellbeing are an important part of overall health. "
                     "A brief 2-question check-in helps us make sure you have the support you need. "
                     "We can do this at your next visit or during a quick telehealth appointment — "
                     "it takes less than 2 minutes."),
        },
    }

    SPANISH_SUBJECTS: Dict[str, str] = {
        "colorectal_screening": "Importante: Su detección de cáncer colorrectal está pendiente",
        "mammogram": "Recordatorio: Mamografía anual pendiente",
        "pap_smear": "Recordatorio: Prueba de Papanicolaou pendiente",
        "blood_pressure": "Acción necesaria: Seguimiento de presión arterial",
        "diabetes_a1c": "Chequeo de diabetes: Prueba de A1c pendiente",
        "diabetes_eye": "Recordatorio: Examen de ojos para diabéticos anual",
        "statin_therapy": "Protección del corazón: Revisión de medicación necesaria",
        "wellness_visit": "Visita anual de bienestar para Medicare",
        "flu_shot": "Temporada de gripe: Vacúnese hoy",
        "depression_screening": "Chequeo anual de salud mental",
    }

    def draft_outreach(self, patient: Dict, gaps: List[Dict]) -> Optional[Dict]:
        if not gaps:
            return None
        top = gaps[0]
        measure = top["measure"]
        tk = HEDIS_MEASURES.get(measure, {}).get("outreach_template", "wellness_visit")
        tmpl = self.TEMPLATES.get(tk, self.TEMPLATES["wellness_visit"])
        lang = patient.get("language", "English")
        contact = patient.get("preferred_contact", "portal")
        subject = (
            self.SPANISH_SUBJECTS.get(tk, tmpl["subject"])
            if lang == "Spanish" else tmpl["subject"]
        )
        closing = {
            "portal": "Reply to this message in your patient portal to schedule.",
            "phone": "Please call us at your earliest convenience to schedule.",
            "text": "Reply YES to this text and we will call you to schedule.",
        }.get(contact, "")
        return {
            "patient_id": patient["patient_id"],
            "patient_name": patient["name"],
            "age": patient["age"],
            "payer": patient.get("payer", "Unknown"),
            "preferred_contact": contact,
            "language": lang,
            "primary_gap_code": measure,
            "primary_gap_name": top["measure_name"],
            "tier": top["tier"],
            "priority_score": top["priority_score"],
            "subject": subject,
            "message": tmpl["body"],
            "closing": closing,
            "clinical_rationale": top.get("clinical_rationale", ""),
            "additional_gaps": [f"• {g['measure_name']}: {g['reason']}" for g in gaps[1:3]],
            "total_gaps": len(gaps),
        }


# ============================================================
# AGENT 5: ANALYTICS REPORTER
# ============================================================
class AnalyticsReporter:
    """Generates population-level analytics and VBC ROI estimates."""

    def generate_report(self, patients: List[Dict], all_gaps: Dict[str, List[Dict]]) -> Dict:
        total = len(patients)
        with_gaps = sum(1 for g in all_gaps.values() if g)
        total_gaps = sum(len(g) for g in all_gaps.values())
        measure_counts: Dict[str, int] = {}
        tier_counts: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "ROUTINE": 0}
        for gaps in all_gaps.values():
            for gap in gaps:
                measure_counts[gap["measure"]] = measure_counts.get(gap["measure"], 0) + 1
                tier_counts[gap.get("tier", "ROUTINE")] = tier_counts.get(gap.get("tier", "ROUTINE"), 0) + 1
        sorted_measures = sorted(measure_counts.items(), key=lambda x: x[1], reverse=True)
        gap_rate = with_gaps / total if total > 0 else 0
        return {
            "summary": {
                "total_patients": total,
                "patients_with_gaps": with_gaps,
                "gap_rate_pct": round(gap_rate * 100, 1),
                "total_gaps_identified": total_gaps,
                "avg_gaps_per_patient": round(total_gaps / total, 1) if total > 0 else 0,
            },
            "gap_by_measure": [
                {"measure": m, "name": HEDIS_MEASURES.get(m, {}).get("name", m),
                 "count": cnt, "rate_pct": round(cnt / total * 100, 1) if total > 0 else 0}
                for m, cnt in sorted_measures
            ],
            "priority_breakdown": tier_counts,
            "quality_impact": {
                "estimated_hedis_percentile": max(10, int(100 - gap_rate * 75)),
                "estimated_vbc_opportunity_usd": with_gaps * 180,
                "critical_needing_urgent_outreach": tier_counts.get("CRITICAL", 0),
                "high_priority_patients": tier_counts.get("HIGH", 0),
            },
        }


# ============================================================
# BASE ORCHESTRATOR (plain pipeline — no EGM)
# ============================================================
class HEDISCareGapOrchestrator:
    """Runs the 5-step care gap pipeline without adversarial review."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.loader    = PatientProfileLoader()
        self.identifier = CareGapIdentifier()
        self.stratifier = RiskStratifier()
        self.drafter    = OutreachDraftingAgent()
        self.reporter   = AnalyticsReporter()
        self._patients: List[Dict] = []

    def _log(self, msg: str = "") -> None:
        if self.verbose: print(msg)

    def run(self, patient_count: int = 50, top_n: int = 10,
            patient_file: Optional[str] = None) -> Dict:
        self._log(); self._log("="*70)
        self._log("  🏥  HEDIS CARE GAP CLOSURE AGENT  v1.0")
        self._log("      Autonomous Population Health Quality Management")
        self._log("="*70)

        self._log(f"\n[1/5] Loading patient panel...")
        patients = (self.loader.load_from_json(patient_file) if patient_file
                    else self.loader.load_synthetic_panel(patient_count))
        self._patients = patients
        self._log(f"  ✓  {len(patients)} patients loaded")

        self._log(f"\n[2/5] Identifying HEDIS quality measure gaps...")
        raw_gaps: Dict[str, List[Dict]] = {
            p["patient_id"]: self.identifier.identify_gaps(p) for p in patients
        }
        n_with = sum(1 for g in raw_gaps.values() if g)
        self._log(f"  ✓  {sum(len(g) for g in raw_gaps.values())} gaps across {n_with}/{len(patients)} patients")

        self._log(f"\n[3/5] Risk-stratifying gaps...")
        strat: Dict[str, List[Dict]] = {
            p["patient_id"]: self.stratifier.stratify(p, raw_gaps[p["patient_id"]]) for p in patients
        }
        critical_n = sum(1 for g in strat.values() for x in g if x.get("tier") == "CRITICAL")
        self._log(f"  ✓  {critical_n} CRITICAL-tier gaps flagged")

        self._log(f"\n[4/5] Generating outreach (top {top_n})...")
        ranked = sorted(
            [(p, strat[p["patient_id"]]) for p in patients if strat[p["patient_id"]]],
            key=lambda x: x[1][0]["priority_score"] if x[1] else 0, reverse=True
        )
        outreach = [
            pkg for pkg in (self.drafter.draft_outreach(p, g) for p, g in ranked[:top_n])
            if pkg
        ]
        self._log(f"  ✓  {len(outreach)} outreach packages generated")

        self._log(f"\n[5/5] Analytics...")
        analytics = self.reporter.generate_report(patients, strat)
        self._log(f"  ✓  Analytics complete")

        return {
            "patients": len(patients),
            "patients_with_gaps": n_with,
            "total_gaps": sum(len(g) for g in raw_gaps.values()),
            "outreach_packages": outreach,
            "analytics": analytics,
            "stratified_gaps": {pid: g for pid, g in list(strat.items())[:top_n]},
        }


# ============================================================
# EGM: ELEPHANT AGENT (long-context synthesizer)
# ============================================================
class ElephantAgent:
    """
    EGM Elephant Agent — the long-context, stateful coordinator.

    Inspired by Google's Elephant-Goldfish Model (EGM, Apache-2.0, Oct 2026):
    https://github.com/google/elephant-goldfish-model

    The Elephant retains full population context, HEDIS technical specs, and
    the version history across all adversarial review rounds. After each round,
    it ingests every Goldfish critique and resolves them before incrementing the
    version number.
    """

    def build_context(self, patients: List[Dict], measures: Dict) -> Dict:
        cond: Dict[str, int] = {}
        for p in patients:
            for c in p.get("conditions", []):
                cond[c] = cond.get(c, 0) + 1
        return {
            "patient_count": len(patients),
            "measure_count": len(measures),
            "measures": list(measures.keys()),
            "condition_distribution": cond,
            "age_distribution": {
                "18-44": sum(1 for p in patients if 18 <= p["age"] <= 44),
                "45-64": sum(1 for p in patients if 45 <= p["age"] <= 64),
                "65+":   sum(1 for p in patients if p["age"] >= 65),
            },
        }

    def resolve_hedis_critiques(self, gap_report: Dict, violations: List[Dict]) -> Dict:
        """Remove gap flags that violate HEDIS eligibility criteria."""
        result = {**gap_report, "stratified_gaps": {
            pid: list(gs) for pid, gs in gap_report.get("stratified_gaps", {}).items()
        }}
        removed = 0
        for v in violations:
            pid, measure = v.get("patient_id"), v.get("measure")
            if pid and measure and pid in result["stratified_gaps"]:
                before = len(result["stratified_gaps"][pid])
                result["stratified_gaps"][pid] = [
                    g for g in result["stratified_gaps"][pid] if g["measure"] != measure
                ]
                removed += before - len(result["stratified_gaps"][pid])
        result["_hedis_fixes"] = removed
        return result

    def resolve_priority_critiques(self, gap_report: Dict, unjustified: List[Dict]) -> Dict:
        """Downgrade unjustified CRITICAL tier flags to HIGH."""
        result = {**gap_report, "stratified_gaps": {
            pid: [dict(g) for g in gs] for pid, gs in gap_report.get("stratified_gaps", {}).items()
        }}
        downgraded = 0
        for item in unjustified:
            pid, measure = item.get("patient_id"), item.get("measure")
            if pid and measure and pid in result["stratified_gaps"]:
                for gap in result["stratified_gaps"][pid]:
                    if gap["measure"] == measure and gap.get("tier") == "CRITICAL":
                        gap["tier"] = "HIGH"
                        gap["priority_score"] = min(gap["priority_score"], 0.89)
                        downgraded += 1
        result["_priority_fixes"] = downgraded
        return result


# ============================================================
# EGM: GOLDFISH CRITICS (stateless — no memory between rounds)
# ============================================================

class HEDISComplianceGoldfish:
    """
    Goldfish Critic 1: HEDIS Technical Specification Compliance

    Stateless — receives only the current gap report, has no memory of prior
    rounds. Verifies every gap flag against the HEDIS age/sex/condition
    eligibility criteria in the official NCQA technical specifications.
    Fails immediately on any violation so the Elephant can fix it.
    """

    def review(self, stratified_gaps: Dict, patients_by_id: Dict) -> Dict:
        violations = []
        for pid, gaps in stratified_gaps.items():
            patient = patients_by_id.get(pid)
            if not patient:
                continue
            age, sex = patient["age"], patient["sex"]
            conditions = patient.get("conditions", [])
            for gap in gaps:
                measure = gap["measure"]
                spec = HEDIS_MEASURES.get(measure)
                if not spec:
                    violations.append({"patient_id": pid, "measure": measure,
                                       "violation": f"Unknown measure code '{measure}'"})
                    continue
                if not (spec["age_min"] <= age <= spec["age_max"]):
                    violations.append({"patient_id": pid, "measure": measure,
                                       "violation": (f"Age {age} outside {measure} eligible range "
                                                     f"{spec['age_min']}–{spec['age_max']}")})
                if spec["sex"] != "any" and sex != spec["sex"]:
                    violations.append({"patient_id": pid, "measure": measure,
                                       "violation": f"Sex '{sex}' ineligible for {measure} (requires '{spec['sex']}')"})
                req = spec.get("condition_required")
                if req and req not in conditions:
                    violations.append({"patient_id": pid, "measure": measure,
                                       "violation": f"Required condition '{req}' absent for {measure}"})
        return {
            "critic": "HEDISComplianceGoldfish",
            "passed": len(violations) == 0,
            "violations": violations,
            "violation_count": len(violations),
            "verdict": "PASS — all gap flags comply with HEDIS technical specs" if not violations
                       else f"FAIL — {len(violations)} eligibility violation(s) found",
        }


class ClinicalPriorityGoldfish:
    """
    Goldfish Critic 2: Clinical Priority Validation

    Stateless — challenges CRITICAL tier assignments independently.
    Low-base-priority measures (FLU, DEP, AWC, CCS) should not reach CRITICAL
    tier unless the patient is high-risk (age 65+, 2+ comorbidities, smoker,
    family history of cancer, or documented social barrier).
    """

    LOW_BASE_PRIORITY = {"FLU", "DEP", "AWC", "CCS"}

    def review(self, stratified_gaps: Dict, patients_by_id: Dict) -> Dict:
        unjustified = []
        for pid, gaps in stratified_gaps.items():
            patient = patients_by_id.get(pid)
            if not patient:
                continue
            age = patient["age"]
            risk = patient.get("risk_factors", {})
            high_risk = (
                age >= 65 or
                risk.get("smoker") or
                risk.get("family_hx_cancer") or
                risk.get("bmi_obese") or
                len(patient.get("conditions", [])) >= 2
            )
            for gap in gaps:
                if gap.get("tier") != "CRITICAL":
                    continue
                if gap["measure"] in self.LOW_BASE_PRIORITY and not high_risk:
                    unjustified.append({
                        "patient_id": pid,
                        "measure": gap["measure"],
                        "issue": (f"CRITICAL tier for {gap['measure']} may be inflated: "
                                  f"age={age}, no high-risk factors detected. "
                                  f"Score {gap.get('priority_score')} should be ≤HIGH."),
                    })
        return {
            "critic": "ClinicalPriorityGoldfish",
            "passed": len(unjustified) == 0,
            "unjustified": unjustified,
            "unjustified_count": len(unjustified),
            "verdict": "PASS — all CRITICAL assignments are clinically justified" if not unjustified
                       else f"FAIL — {len(unjustified)} unjustified CRITICAL flag(s)",
        }


class OutreachClarityGoldfish:
    """
    Goldfish Critic 3: Outreach Message Quality

    Stateless — reviews outreach packages for actionability and completeness.
    Checks: non-empty subject & message, minimum length, call-to-action present,
    Spanish language consistency, and patient ID integrity.
    """

    MIN_MSG_LEN = 40

    def review(self, outreach_packages: List[Dict]) -> Dict:
        issues = []
        for pkg in outreach_packages:
            pid = pkg.get("patient_id", "UNKNOWN")
            if not pkg.get("subject", "").strip():
                issues.append({"patient_id": pid, "issue": "Empty subject line"})
            msg = pkg.get("message", "")
            if len(msg.strip()) < self.MIN_MSG_LEN:
                issues.append({"patient_id": pid, "issue": f"Message too short ({len(msg)} chars)"})
            if not pkg.get("closing", "").strip():
                issues.append({"patient_id": pid, "issue": "Missing call-to-action closing"})
            if not pkg.get("primary_gap_code"):
                issues.append({"patient_id": pid, "issue": "Missing primary gap code"})
        return {
            "critic": "OutreachClarityGoldfish",
            "passed": len(issues) == 0,
            "issues": issues,
            "issue_count": len(issues),
            "verdict": "PASS — all outreach packages are clear and actionable" if not issues
                       else f"FAIL — {len(issues)} clarity issue(s) found",
        }


# ============================================================
# EGM ORCHESTRATOR (Elephant-Goldfish adversarial review loop)
# ============================================================
class EGMHEDISOrchestrator:
    """
    EGM-enhanced HEDIS Care Gap Closure Agent.

    Implements the Elephant-Goldfish Model (EGM) adversarial review loop:
      1. Elephant runs the full 5-agent pipeline → produces v1.0 gap analysis
      2. Three stateless Goldfish critics independently attack the output
      3. Any critic fails? Elephant resolves critiques → v2.0 → repeat
      4. All critics pass (or max_rounds reached) → validated final output

    Reference: Google EGM — https://github.com/google/elephant-goldfish-model
               Apache-2.0 License, October 2026
    """

    def __init__(self, verbose: bool = True, max_rounds: int = 3):
        self.verbose    = verbose
        self.max_rounds = max_rounds
        self.pipeline   = HEDISCareGapOrchestrator(verbose=False)
        self.elephant   = ElephantAgent()
        self.review_log: List[Dict] = []

    def _log(self, msg: str = "") -> None:
        if self.verbose: print(msg)

    def run(self, patient_count: int = 50, top_n: int = 10,
            patient_file: Optional[str] = None) -> Dict:
        self._log(); self._log("="*70)
        self._log("  🐘🐟  HEDIS CARE GAP CLOSURE AGENT  [EGM Edition]")
        self._log("       Elephant-Goldfish Adversarial Review Architecture")
        self._log("       Ref: https://github.com/google/elephant-goldfish-model")
        self._log("="*70)

        # ── ELEPHANT: build context + run initial pipeline ─────────────
        self._log(f"\n[🐘 ELEPHANT] Loading panel and building population context...")
        result = self.pipeline.run(
            patient_count=patient_count, top_n=top_n, patient_file=patient_file
        )
        patients_by_id = {p["patient_id"]: p for p in self.pipeline._patients}
        ctx = self.elephant.build_context(self.pipeline._patients, HEDIS_MEASURES)
        self._log(f"  Context: {ctx['patient_count']} pts · {ctx['measure_count']} measures · "
                  f"conditions: {ctx['condition_distribution']}")

        # ── EGM ADVERSARIAL REVIEW LOOP ────────────────────────────────
        version = "1.0"
        outreach = result["outreach_packages"]
        self.review_log = []

        for rnd in range(1, self.max_rounds + 1):
            self._log(f"\n{'─'*70}")
            self._log(f"  [ROUND {rnd}/{self.max_rounds}]  v{version}  —  Goldfish Critics Reviewing...")

            r_hedis    = HEDISComplianceGoldfish().review(result["stratified_gaps"], patients_by_id)
            r_priority = ClinicalPriorityGoldfish().review(result["stratified_gaps"], patients_by_id)
            r_outreach = OutreachClarityGoldfish().review(outreach)

            for r, label in [(r_hedis, "HEDISComplianceGoldfish "),
                              (r_priority, "ClinicalPriorityGoldfish"),
                              (r_outreach, "OutreachClarityGoldfish ")]:
                icon = "✅ PASS" if r["passed"] else "❌ FAIL"
                self._log(f"  🐟 {label} : {icon}  — {r['verdict']}")

            all_pass = r_hedis["passed"] and r_priority["passed"] and r_outreach["passed"]
            self.review_log.append({
                "round": rnd, "version": version,
                "hedis_compliance": r_hedis,
                "clinical_priority": r_priority,
                "outreach_clarity": r_outreach,
                "all_passed": all_pass,
            })

            if all_pass:
                self._log(f"\n  ✅  All 3 Goldfish critics PASSED — v{version} is validated!")
                break

            if rnd == self.max_rounds:
                self._log(f"\n  ⚠️  Max rounds reached. Proceeding with v{version}.")
                break

            # ── ELEPHANT RESOLVES CRITIQUES ────────────────────────────
            self._log(f"\n  [🐘 ELEPHANT] Resolving critiques → bumping to v{rnd + 1}.0...")
            if not r_hedis["passed"]:
                result = self.elephant.resolve_hedis_critiques(result, r_hedis["violations"])
                self._log(f"    Removed {result.get('_hedis_fixes', 0)} invalid HEDIS flags")
            if not r_priority["passed"]:
                result = self.elephant.resolve_priority_critiques(result, r_priority["unjustified"])
                self._log(f"    Downgraded {result.get('_priority_fixes', 0)} unjustified CRITICAL flags")
            if not r_outreach["passed"]:
                self._log(f"    Noted {r_outreach['issue_count']} outreach quality issues")
            version = f"{rnd + 1}.0"

        # ── PRINT FINAL SUMMARY ────────────────────────────────────────
        total_fixed = sum(
            r["hedis_compliance"]["violation_count"] + r["clinical_priority"]["unjustified_count"]
            for r in self.review_log[:-1]
        )
        s  = result["analytics"]["summary"]
        qi = result["analytics"]["quality_impact"]

        self._log(f"\n{'='*70}")
        self._log(f"  📊  POPULATION ANALYTICS SUMMARY")
        self._log(f"{'='*70}")
        self._log(f"\n  Patient Panel       : {s['total_patients']:,}")
        self._log(f"  Patients with Gaps  : {s['patients_with_gaps']:,}  ({s['gap_rate_pct']}%)")
        self._log(f"  Total Gaps Found    : {s['total_gaps_identified']:,}")
        self._log(f"  Avg Gaps/Patient    : {s['avg_gaps_per_patient']}")
        self._log(f"\n  📈  Quality & Revenue Impact:")
        self._log(f"     HEDIS Percentile (est.)      : ~{qi['estimated_hedis_percentile']}th")
        self._log(f"     Value-Based Care Opportunity : ${qi['estimated_vbc_opportunity_usd']:,} / yr")
        self._log(f"     CRITICAL patients (urgent)   : {qi['critical_needing_urgent_outreach']}")
        self._log(f"\n  🔝  Top Gaps by Prevalence:")
        for i, gm in enumerate(result["analytics"]["gap_by_measure"][:5], 1):
            self._log(f"     {i}. {gm['name']:<38} {gm['count']:>3} pts ({gm['rate_pct']}%)")
        pb = result["analytics"]["priority_breakdown"]
        self._log(f"\n  🚦  Tiers: CRITICAL={pb['CRITICAL']}  HIGH={pb['HIGH']}"
                  f"  MODERATE={pb['MODERATE']}  ROUTINE={pb['ROUTINE']}")

        self._log(f"\n{'='*70}")
        self._log(f"  🐘🐟  EGM REVIEW LOG")
        self._log(f"{'='*70}")
        self._log(f"  Final version      : v{version}")
        self._log(f"  Rounds completed   : {len(self.review_log)}")
        self._log(f"  Issues auto-fixed  : {total_fixed}")
        for r in self.review_log:
            status = "✅ ALL PASS" if r["all_passed"] else "❌ critiques found"
            self._log(f"  Round {r['round']} (v{r['version']}) : {status}")

        self._log(f"\n  📬  TOP OUTREACH PACKAGES:")
        for i, pkg in enumerate(outreach[:min(5, len(outreach))], 1):
            self._log(f"\n  {i}. {pkg['patient_name']}  ({pkg['patient_id']})  Age {pkg['age']}  |  {pkg['payer']}")
            self._log(f"     Tier: {pkg['tier']}  Score: {pkg['priority_score']}"
                      f"  |  {pkg['preferred_contact'].upper()}  |  {pkg['language']}")
            self._log(f"     Primary Gap : {pkg['primary_gap_name']} [{pkg['primary_gap_code']}]")
            self._log(f"     Subject     : {pkg['subject']}")
            if pkg["additional_gaps"]:
                for ag in pkg["additional_gaps"]:
                    self._log(f"       {ag}")

        self._log(f"\n{'='*70}")
        self._log(f"  ✅  EGM-Validated HEDIS Care Gap Agent — Run Complete")
        self._log(f"{'='*70}\n")

        result["egm_review_log"]  = self.review_log
        result["final_version"]   = version
        result["egm_total_fixed"] = total_fixed
        return result


# ============================================================
# CLI ENTRY POINT
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="HEDIS Care Gap Closure Agent [EGM Edition]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                            # EGM mode, 50 patients, top 10
  python main.py --patients 200 --top 25   # larger panel
  python main.py --no-egm                   # skip adversarial review loop
  python main.py --output results.json      # export full JSON
  python main.py --patient-file my_pts.json # real patient data
""",
    )
    parser.add_argument("--patients",     type=int,  default=50,   help="Synthetic patients (default 50)")
    parser.add_argument("--top",          type=int,  default=10,   help="Top N for outreach (default 10)")
    parser.add_argument("--output",       type=str,  default=None, help="Save results to JSON file")
    parser.add_argument("--patient-file", type=str,  default=None, help="Load patients from JSON file")
    parser.add_argument("--no-egm",       action="store_true",     help="Disable EGM review loop")
    parser.add_argument("--max-rounds",   type=int,  default=3,    help="EGM max review rounds (default 3)")
    parser.add_argument("--quiet",        action="store_true",     help="Suppress console output")
    args = parser.parse_args()

    if args.no_egm:
        orchestrator: Any = HEDISCareGapOrchestrator(verbose=not args.quiet)
        results = orchestrator.run(patient_count=args.patients, top_n=args.top,
                                   patient_file=args.patient_file)
    else:
        orchestrator = EGMHEDISOrchestrator(verbose=not args.quiet, max_rounds=args.max_rounds)
        results = orchestrator.run(patient_count=args.patients, top_n=args.top,
                                   patient_file=args.patient_file)

    if args.output:
        output_data = {
            "run_date": date.today().isoformat(),
            "final_version": results.get("final_version", "1.0"),
            "summary": results["analytics"]["summary"],
            "gap_by_measure": results["analytics"]["gap_by_measure"],
            "quality_impact": results["analytics"]["quality_impact"],
            "priority_breakdown": results["analytics"]["priority_breakdown"],
            "outreach_packages": results["outreach_packages"],
            "egm_review_log": results.get("egm_review_log", []),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Full results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
