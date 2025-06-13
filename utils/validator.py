import os
import sqlite3
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict
from werkzeug.security import generate_password_hash, check_password_hash

# --- Database Setup ---
DB_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            auth_provider TEXT DEFAULT 'email'
        )
    ''')
    conn.commit()
    conn.close()

# --- Validation Challenges List (Comprehensive) ---
validation_challenges = {
    "A. Regulatory & Governance Challenges": [
        "Misalignment with PRA SS1/23 expectations",
        "Failure to demonstrate forward-looking information integration",
        "Incomplete documentation on model assumptions",
        "Lack of evidence of internal challenge/independent validation",
        "Weak justification for staging thresholds (SICR triggers)",
        "Inadequate validation of staging logic and override policy",
        "Insufficient backtesting of modelled PDs vs actual defaults",
        "Absence of model governance traceability and sign-offs",
        "No model risk tiering and prioritization defined",
        "Failure to separate IFRS 9 and IRB objectives clearly"
    ],
    "B. Data & Input Issues": [
        "Missing or incomplete historical data for key drivers",
        "Data not aligned with reporting/accounting granularity",
        "Biased datasets (e.g. selection bias due to charge-offs)",
        "Ignoring payment holiday or forbearance periods",
        "Insufficient cure window post-default",
        "Inadequate performance window definitions",
        "No vintage or cohort tracking in raw data",
        "Errors in macroeconomic input mapping (lag, level)",
        "Inconsistent treatment of revolving vs transactors",
        "Incorrect use of behavioural data (e.g. utilization)"
    ],
    "C. Model Methodology Challenges": [
        "Logistic regression lacks explanatory power (low Gini)",
        "Model performance degrades materially across vintages",
        "Lifetime PD extension lacks statistical justification",
        "Use of survival model lacks cure-adjusted hazard rates",
        "No explicit error bounds/confidence intervals provided",
        "Ignoring maturity effect on credit card behaviour",
        "Transition matrix method misapplied or unstable",
        "Spurious correlation between macro drivers and defaults",
        "Overfitting in machine learning models (XGBoost, RF)",
        "Ignoring account seasoning in PD maturity profile"
    ],
    "D. Assumptions & Simplifications": [
        "Lifetime PD extrapolation is linear/unrealistic",
        "Too few macroeconomic scenarios used",
        "Macro scenarios not severe enough for downturn risk",
        "No sensitivity analysis for assumptions",
        "Ignoring behavioural changes due to pricing/APR changes",
        "Static exposure profiles assumed over life",
        "No borrower-level dynamic modelling",
        "Lack of justification for segmentation rules",
        "Default definition differs from policy/international norms",
        "Proxy used for default where DPD not reliable"
    ],
    "E. Macroeconomic Integration Weaknesses": [
        "Weak or no statistical linkage between macro and PD",
        "Scenario expansion method (e.g., delta shift) is overly simplistic",
        "Ignoring interaction effects among macro variables",
        "No validation of macroeconomic overlay adjustments",
        "Manual scenario overrides not tracked or justified",
        "Misaligned macroeconomic scenario time horizon",
        "Failure to document macroeconomic source credibility",
        "Over-dependence on GDP/unemployment without justification",
        "No base-vs-downturn impact comparison",
        "Model fails to show responsiveness to stressed scenarios"
    ],
    "F. Portfolio-Specific Limitations": [
        "Model does not differentiate between transactors and revolvers",
        "Lack of segmentation for teaser-rate or 0% BT products",
        "Fails to model exposure dynamics for dormant/reactivated accounts",
        "Lifetime PD doesn’t reflect prepayment/closure risks",
        "Changes in credit limits and exposure not accounted for",
        "No vintage sensitivity tracking over time",
        "Mismatch between booked exposures and active accounts",
        "Product-specific risk features not captured (e.g. loyalty cards)",
        "Ignoring self-cures and behavioural cyclicality",
        "Model not recalibrated post COVID/shock period"
    ],
    "G. Model Validation Weaknesses": [
        "No independent challenger model developed",
        "Validation sample too short or not representative",
        "Inadequate out-of-time testing",
        "Stability metrics (PSI, CoV) not monitored or reported",
        "KS/AUC not benchmarked to acceptable thresholds",
        "Lack of lift analysis across PD bands",
        "Default prediction errors not diagnosed across segments",
        "PD distributions do not match expected shape",
        "Rejection inference not addressed (origination model reused)",
        "Benchmarking with peer models or vendor models missing"
    ],
    "H. Calibration & Monitoring Issues": [
        "No lifetime calibration back to observed defaults",
        "Transition from 12m to lifetime PD poorly controlled",
        "Calibration overrides undocumented or unjustified",
        "No cap/floor logic on PDs for rare segments",
        "Failure to update model in light of monitoring breaches",
        "Model recalibration frequency unclear or ad hoc",
        "Output drifts without input change not explained",
        "Lack of automated monitoring or alerting system",
        "Model decay not assessed annually",
        "Missing override logs and escalation steps"
    ]
}

# --- User Authentication ---
def signup_user(name: str, email: str, password: str) -> bool:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        c.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (name, email, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_login(email: str, password: str) -> Dict[str, str] | None:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, email, password_hash FROM users WHERE email = ?", (email,))
    user_record = c.fetchone()
    conn.close()
    if user_record and check_password_hash(user_record[2], password):
        return {"name": user_record[0], "email": user_record[1]}
    return None

# --- Validation Logic Example ---
def validate_model_documentation(doc_text: str, code_content: str, validation_level: str = "Standard Review") -> Dict[str, Any]:
    """
    Validates model documentation and code against regulatory and best practice challenges.
    Returns a dict with compliance scores and section breakdowns.
    """
    assessment_results = {
        "overall_score": 0,
        "total_possible_score": 0,
        "risk_rating": "N/A",
        "compliant_count": 0,
        "partially_compliant_count": 0,
        "non_compliant_count": 0,
        "error_count": 0,
        "sections": {},
        "gap_table": [],
        "summary": ""
    }
    # Example: For each section, check if key phrases from challenges are present in doc/code
    for section, challenges in validation_challenges.items():
        section_result = {"compliant": 0, "partial": 0, "non_compliant": 0, "errors": 0, "details": []}
        for challenge in challenges:
            found = (challenge.lower() in doc_text.lower()) or (challenge.lower() in code_content.lower())
            if found:
                section_result["compliant"] += 1
            else:
                section_result["non_compliant"] += 1
                section_result["details"].append(challenge)
        assessment_results["compliant_count"] += section_result["compliant"]
        assessment_results["non_compliant_count"] += section_result["non_compliant"]
        assessment_results["sections"][section] = section_result
        assessment_results["total_possible_score"] += len(challenges)
        assessment_results["overall_score"] += section_result["compliant"]
        if section_result["non_compliant"] > 0:
            assessment_results["gap_table"].append({"section": section, "gaps": section_result["details"]})
    # Risk rating logic
    score_pct = (assessment_results["overall_score"] / assessment_results["total_possible_score"]) * 100 if assessment_results["total_possible_score"] > 0 else 0
    if score_pct >= 85:
        assessment_results["risk_rating"] = "Low"
    elif score_pct >= 70:
        assessment_results["risk_rating"] = "Moderate"
    elif score_pct >= 50:
        assessment_results["risk_rating"] = "Elevated"
    else:
        assessment_results["risk_rating"] = "High"
    assessment_results["summary"] = f"Overall Score: {assessment_results['overall_score']}/{assessment_results['total_possible_score']} ({score_pct:.1f}%). Risk Rating: {assessment_results['risk_rating']}. Compliant: {assessment_results['compliant_count']}, Non-Compliant: {assessment_results['non_compliant_count']}."
    return assessment_results
