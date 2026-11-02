# 🏥 HEDIS Care Gap Closure Agent  [EGM Edition]

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-115%20passing-brightgreen)
![Zero Paid APIs](https://img.shields.io/badge/APIs-zero%20paid-orange)
![Agentic AI](https://img.shields.io/badge/agentic-AI-purple)
![EGM](https://img.shields.io/badge/EGM-adversarial%20review-red)

> **Autonomous 5-step AI pipeline that identifies HEDIS quality measure gaps across a patient population, risk-stratifies each gap, and generates personalized care outreach — enhanced with Google's Elephant-Goldfish Model adversarial review loop.**

---

## The Problem

US healthcare practices leave **~$100 billion per year** in value-based care bonuses unrealized because patients fall behind on preventive screenings.

**HEDIS** (Healthcare Effectiveness Data and Information Set) is the standard set of quality measures used by CMS and virtually every commercial insurer to evaluate clinical quality. Poor HEDIS performance means:

- 💸 Withheld bonus payments and quality penalties (often $200–$500 per patient per gap)
- 📉 Lower star ratings that reduce Medicare Advantage premiums
- 😞 Worse patient outcomes from missed early detection

**The real bottleneck isn't clinical knowledge — it's trust and bandwidth.** A single wrong gap flag (e.g., flagging a 43-year-old for colorectal screening, or flagging a male patient for a mammogram) erodes care coordinator trust in the system. This agent uses an adversarial review loop to catch these errors before they reach the queue.

---

## EGM Adversarial Review Architecture

This project integrates **Google's Elephant-Goldfish Model (EGM)** — an adversarial review architecture where a long-context Elephant synthesizer iterates with stateless Goldfish critics until the output meets all quality criteria.

Reference: [https://github.com/google/elephant-goldfish-model](https://github.com/google/elephant-goldfish-model) (Apache-2.0, October 2026)

```
Patient Population Panel
         │
         ▼
[🐘 ELEPHANT] ElephantAgent
         Builds full population context (demographics, conditions,
         age distribution) and runs the complete 5-agent pipeline
         → produces v1.0 gap analysis
         │
         ▼  ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
         │                                              │  (loop while critics fail,
[🐟 GOLDFISH CRITICS]  (3 stateless, independent)      │   max N rounds)
         │                                              │
         ├─ HEDISComplianceGoldfish                     │
         │     Verifies every gap flag against NCQA     │
         │     HEDIS specs: age range, sex eligibility, │
         │     required condition (e.g., no CBP flag    │
         │     without hypertension diagnosis)          │
         │                                              │
         ├─ ClinicalPriorityGoldfish                    │
         │     Challenges unjustified CRITICAL tiers    │
         │     for low-base-priority measures           │
         │     (FLU, DEP, AWC, CCS) where patient is   │
         │     not high-risk                            │
         │                                              │
         └─ OutreachClarityGoldfish                     │
               Reviews outreach packages: non-empty     │
               subject & message, min 40-char message,  │
               call-to-action closing, gap code present  │
                                                        │
         ▼                                              │
  Any critic FAIL? ───────────────────────────────────>┘
  [🐘 ELEPHANT] resolves critiques:
    • Removes invalid HEDIS gap flags
    • Downgrades unjustified CRITICAL → HIGH
    • Version bumps: v1.0 → v2.0 → ...
         │
  All critics PASS or max rounds reached
         │
         ▼
  Validated final output + egm_review_log
```

### What EGM Catches

| Critic | Example Issue Caught |
|--------|---------------------|
| HEDISComplianceGoldfish | 43-year-old flagged for colorectal screening (min age 45) |
| HEDISComplianceGoldfish | Male patient flagged for mammogram (female-only measure) |
| HEDISComplianceGoldfish | CBP flag without hypertension diagnosis |
| ClinicalPriorityGoldfish | Young, healthy patient gets CRITICAL flu shot tier (should be ROUTINE) |
| ClinicalPriorityGoldfish | DEP flagged CRITICAL for low-risk 30-year-old |
| OutreachClarityGoldfish | Outreach subject is empty string |
| OutreachClarityGoldfish | Message body is only 12 characters |
| OutreachClarityGoldfish | Missing call-to-action closing |

---

## 5-Step Agentic Pipeline

```
[Agent 1] PatientProfileLoader
         Loads and normalizes patient data
         (demographics, conditions, last screening dates, risk factors)
         │
         ▼
[Agent 2] CareGapIdentifier
         Evaluates each patient against 10 HEDIS measures:
         COLO · BCS · CCS · CBP · CDC_A1C · CDC_EYE · LDL · AWC · FLU · DEP
         │
         ▼
[Agent 3] RiskStratifier
         Computes a composite priority score (0–1.0) per gap using:
         • HEDIS measure clinical weight
         • Patient-specific risk factors (smoking, obesity, family Hx, SDOH)
         • Comorbidity burden
         • Overdue severity
         Assigns tier: CRITICAL → HIGH → MODERATE → ROUTINE
         │
         ▼
[Agent 4] OutreachDraftingAgent
         Generates personalized messages for the top-N priority patients:
         • Measure-specific clinical messaging
         • Language adaptation (English, Spanish, more)
         • Channel-appropriate closing (portal / phone / text)
         • Surfaces up to 2 additional co-present gaps
         │
         ▼
[Agent 5] AnalyticsReporter
         Population-level analytics:
         • Gap rate by HEDIS measure
         • Priority tier breakdown
         • HEDIS percentile estimate
         • Value-based care revenue opportunity ($)
```

---

## HEDIS Measures Covered

| Code | Measure | Population |
|------|---------|------------|
| COLO | Colorectal Cancer Screening | Adults 45–75 |
| BCS  | Breast Cancer Screening (Mammogram) | Women 50–74 |
| CCS  | Cervical Cancer Screening (Pap) | Women 21–64 |
| CBP  | Controlling High Blood Pressure | HTN patients 18–85 |
| CDC_A1C | Diabetes A1c Testing & Control | Diabetics 18–75 |
| CDC_EYE | Diabetic Retinal Eye Exam | Diabetics 18–75 |
| LDL  | Statin Therapy for CVD | CVD patients 21–85 |
| AWC  | Annual Wellness Visit | Medicare 65+ |
| FLU  | Influenza Vaccination | Adults 50+ |
| DEP  | Depression Screening (PHQ-2/PHQ-9) | All adults 18+ |

---

## Install

```bash
# Clone the repo
git clone https://github.com/ai-summer2025/hedis-care-gap-agent.git
cd hedis-care-gap-agent

# No external dependencies needed — pure Python stdlib
python --version  # Requires Python 3.9+
```

**Requirements:** Python 3.9+ and nothing else. No `pip install` needed. Zero paid APIs.

---

## Usage

### Quick start (50 synthetic patients, EGM mode)
```bash
python main.py
```

### Larger panel with JSON output
```bash
python main.py --patients 200 --top 20 --output gaps_report.json
```

### Plain pipeline mode (no adversarial review)
```bash
python main.py --no-egm --patients 100
```

### Configure EGM review rounds
```bash
python main.py --max-rounds 5 --patients 100 --top 15
```

### Quiet mode (for CI / scripted pipelines)
```bash
python main.py --patients 100 --quiet --output results.json
```

### Options
```
--patients     N    Number of synthetic patients to generate (default: 50)
--top          N    Generate outreach for top N priority patients (default: 10)
--output       FILE Save full results to a JSON file
--patient-file FILE Load patients from a JSON file instead of synthetic
--no-egm            Disable EGM adversarial review loop
--max-rounds   N    EGM max review rounds (default: 3)
--quiet             Suppress console output (useful for scripting)
```

---

## Sample Output (EGM Edition)

```
======================================================================
  🐘🐟  HEDIS CARE GAP CLOSURE AGENT  [EGM Edition]
       Elephant-Goldfish Adversarial Review Architecture
       Ref: https://github.com/google/elephant-goldfish-model
======================================================================

[🐘 ELEPHANT] Loading panel and building population context...
  Context: 50 pts · 10 measures · conditions: {'diabetes': 12, ...}

──────────────────────────────────────────────────────────────────────
  [ROUND 1/3]  v1.0  —  Goldfish Critics Reviewing...
  🐟 HEDISComplianceGoldfish  : ✅ PASS  — all gap flags comply with HEDIS specs
  🐟 ClinicalPriorityGoldfish : ✅ PASS  — all CRITICAL assignments justified
  🐟 OutreachClarityGoldfish  : ✅ PASS  — all outreach packages clear

  ✅  All 3 Goldfish critics PASSED — v1.0 is validated!

======================================================================
  📊  POPULATION ANALYTICS SUMMARY
======================================================================

  Patient Panel       : 50
  Patients with Gaps  : 47  (94.0%)
  Total Gaps Found    : 187
  Avg Gaps/Patient    : 3.7

  📈  Quality & Revenue Impact:
     HEDIS Percentile (est.)      : ~29th
     Value-Based Care Opportunity : $8,460 / yr
     CRITICAL patients (urgent)   : 23

  🔝  Top Gaps by Prevalence:
     1. Depression Screening                    46 pts (92.0%)
     2. Influenza Vaccination                   38 pts (76.0%)
     3. Colorectal Cancer Screening             32 pts (64.0%)
     4. Annual Wellness Visit                   21 pts (42.0%)
     5. Controlling High Blood Pressure         18 pts (36.0%)

  🚦  Tiers: CRITICAL=23  HIGH=41  MODERATE=67  ROUTINE=56

======================================================================
  🐘🐟  EGM REVIEW LOG
======================================================================
  Final version      : v1.0
  Rounds completed   : 1
  Issues auto-fixed  : 0
  Round 1 (v1.0) : ✅ ALL PASS
```

---

## Why This Matters

### For Practices on Value-Based Contracts
Every closed care gap translates directly to quality bonus payments. A 2,000-patient practice with 40% gap rate has ~800 patients needing outreach — this agent processes them in seconds and identifies the 50 that are CRITICAL priority. The EGM review loop ensures the care coordinator never sees a gap flag that violates HEDIS eligibility — preserving trust in the tool.

### For Care Coordinators
Instead of working from a flat spreadsheet list, coordinators get a risk-stratified queue with personalized outreach messages ready to send, in the patient's preferred language, via their preferred contact channel. The adversarial validation layer means fewer embarrassing errors (wrong age, wrong sex, missing diagnosis).

### For Health Systems
Population-level analytics surface which HEDIS measures are dragging down the star rating and exactly how many patients need intervention — with an estimated dollar impact to make the business case for action.

### The Agentic Difference + EGM
This is not a dashboard that shows a care manager a list. It is an **autonomous agent** that:
1. Identifies every gap for every patient simultaneously
2. Weighs each gap against that patient's specific risk profile
3. Decides which patients and gaps to prioritize
4. Drafts the actual message — not a template placeholder, but a ready-to-send communication
5. **EGM: Validates every flag and priority against NCQA HEDIS technical specifications**
6. **EGM: Auto-corrects eligibility violations and unjustified CRITICAL tiers**
7. Presents the practice with a ranked, validated action queue

---

## Project Structure

```
hedis-care-gap-agent/
├── main.py              # Full EGM-enhanced agent pipeline (all agents + CLI)
├── test_main.py         # 115 unit tests covering all agents including EGM
├── requirements.txt     # No external dependencies
├── .gitignore
└── README.md
```

---

## Extending This Project

- **Connect to a real EHR**: Replace `PatientProfileLoader.load_from_json()` with a FHIR R4 API call (HL7 FHIR is free/open standard)
- **Add more HEDIS measures**: Drop new entries into `HEDIS_MEASURES` dict and add a method to `CareGapIdentifier`
- **Add more Goldfish critics**: Implement a `SDOHBarrierGoldfish` to flag patients with transportation/cost barriers who need special outreach
- **Add more languages**: Extend `SPANISH_SUBJECTS` in `OutreachDraftingAgent` for any language
- **Add NLP layer**: Pipe patient notes through a free local model (Ollama) to extract unstructured screening history
- **Add CMS Star Rating predictor**: Use the gap rates to estimate star rating movement

---

## Research Sources

- NCQA HEDIS Technical Specifications 2025/2026: https://www.ncqa.org/hedis/
- Google Elephant-Goldfish Model (EGM): https://github.com/google/elephant-goldfish-model (Apache-2.0)
- CMS Star Ratings methodology: https://www.cms.gov/medicare/health-plans/quality-improvement-program
- AHRQ: Closing care gaps in value-based care (2025)
- AMA: Prior Auth & Quality Reporting Burden Survey (2025) — physicians spend 13+ hrs/week on quality admin
- HFMA (2026): "AI Evolution of Denials Management" — $262B in annual denied healthcare revenue

---

## License

MIT License — free to use, fork, and build on.

```
Copyright (c) 2026 ai-summer2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
