#!/usr/bin/env python3
"""
Quick Model Testing Script
===========================
Run predictions directly against the AI service without using the UI.

Usage:
    python test_predictions.py                    # Run all test cases
    python test_predictions.py --case 1           # Run specific case (1-6)
    python test_predictions.py --custom           # Run with custom data
"""

import requests
import json
import sys
from datetime import datetime

AI_SERVICE_URL = "http://localhost:8000"

# Pre-defined test cases with expected outcomes
TEST_CASES = {
    1: {
        "name": "Low Risk - Young, Matched Sibling",
        "expected": "Low",
        "data": {
            "age_at_hct": 25,
            "donor_age": 28,
            "year_hct": 2023,
            "prim_disease_hct": "AML",
            "dri_score": "Low",
            "donor_related": "Sibling",
            "conditioning_intensity": "RIC",
            "gvhd_proph": "TAC+MTX",
            "hla_high_res_8": 8,
            "karnofsky_score": 90,
            "comorbidity_score": 0,
            "tbi_status": "No TBI",
            "race_group": "White",
            "ethnicity": "Not Hispanic or Latino",
            "sex_match": "M-M"
        }
    },
    2: {
        "name": "Medium Risk - Middle Age, Unrelated Donor",
        "expected": "Medium",
        "data": {
            "age_at_hct": 45,
            "donor_age": 35,
            "year_hct": 2022,
            "prim_disease_hct": "MDS",
            "dri_score": "Intermediate",
            "donor_related": "Unrelated",
            "conditioning_intensity": "MAC",
            "gvhd_proph": "TAC+MTX",
            "hla_high_res_8": 7,
            "karnofsky_score": 80,
            "comorbidity_score": 2,
            "tbi_status": "TBI",
            "race_group": "White",
            "ethnicity": "Not Hispanic or Latino",
            "sex_match": "F-M",
            "cmv_status": "+/-"
        }
    },
    3: {
        "name": "High Risk - Elderly, Mismatched, Advanced Disease",
        "expected": "High",
        "data": {
            "age_at_hct": 68,
            "donor_age": 45,
            "year_hct": 2021,
            "prim_disease_hct": "AML",
            "dri_score": "High",
            "donor_related": "Unrelated",
            "conditioning_intensity": "MAC",
            "gvhd_proph": "Other",
            "hla_high_res_8": 5,
            "karnofsky_score": 60,
            "comorbidity_score": 5,
            "tbi_status": "TBI",
            "race_group": "Black or African-American",
            "ethnicity": "Not Hispanic or Latino",
            "sex_match": "F-M",
            "cmv_status": "+/+",
            "cardiac": "Y",
            "diabetes": "Y",
            "pulm_moderate": "Y",
            "renal_issue": "Y"
        }
    },
    4: {
        "name": "Pediatric - Very Young Patient",
        "expected": "Low",
        "data": {
            "age_at_hct": 8,
            "donor_age": 35,
            "year_hct": 2023,
            "prim_disease_hct": "ALL",
            "dri_score": "Low",
            "donor_related": "Sibling",
            "conditioning_intensity": "MAC",
            "gvhd_proph": "TAC+MTX",
            "hla_high_res_8": 8,
            "karnofsky_score": 100,
            "comorbidity_score": 0,
            "tbi_status": "TBI",
            "race_group": "Asian",
            "ethnicity": "Not Hispanic or Latino",
            "sex_match": "M-M"
        }
    },
    5: {
        "name": "High Risk - Multiple Comorbidities",
        "expected": "High",
        "data": {
            "age_at_hct": 55,
            "donor_age": 50,
            "year_hct": 2020,
            "prim_disease_hct": "Other Leukemia",
            "dri_score": "Very High",
            "donor_related": "Unrelated",
            "conditioning_intensity": "MAC",
            "gvhd_proph": "Other",
            "hla_high_res_8": 4,
            "karnofsky_score": 50,
            "comorbidity_score": 7,
            "tbi_status": "TBI",
            "race_group": "White",
            "ethnicity": "Hispanic or Latino",
            "sex_match": "F-M",
            "cmv_status": "+/+",
            "cardiac": "Y",
            "arrhythmia": "Y",
            "diabetes": "Y",
            "hepatic_mild": "Y",
            "hepatic_severe": "Y",
            "obesity": "Y",
            "pulm_severe": "Y",
            "renal_issue": "Y"
        }
    },
    6: {
        "name": "Cord Blood Transplant",
        "expected": "Medium",
        "data": {
            "age_at_hct": 35,
            "donor_age": 0,
            "year_hct": 2022,
            "prim_disease_hct": "AML",
            "dri_score": "Intermediate",
            "graft_type": "Cord blood",
            "conditioning_intensity": "RIC",
            "gvhd_proph": "CSA+MMF",
            "hla_high_res_8": 5,
            "karnofsky_score": 85,
            "comorbidity_score": 1,
            "tbi_status": "No TBI",
            "race_group": "More than one race",
            "ethnicity": "Not Hispanic or Latino"
        }
    }
}


def check_service():
    """Check if AI service is running."""
    try:
        resp = requests.get(f"{AI_SERVICE_URL}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


def run_prediction(patient_data: dict) -> dict:
    """Run a single prediction against the AI service."""
    resp = requests.post(
        f"{AI_SERVICE_URL}/predict",
        json=patient_data,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def print_result(case_num: int, case_info: dict, result: dict):
    """Pretty print prediction result."""
    expected = case_info["expected"]
    actual = result.get("risk_category", "Unknown")
    probability = result.get("event_probability", 0) * 100
    
    match_status = "✅" if expected == actual else "❌"
    
    print(f"\n{'='*60}")
    print(f"Case {case_num}: {case_info['name']}")
    print(f"{'='*60}")
    print(f"  Expected:    {expected}")
    print(f"  Actual:      {actual} {match_status}")
    print(f"  Probability: {probability:.1f}%")
    print(f"  Confidence:  {result.get('confidence_level', 'N/A')}")
    
    if "survival_at_days" in result:
        print(f"\n  Survival Estimates:")
        for days, prob in result["survival_at_days"].items():
            print(f"    {days} days: {prob*100:.1f}%")
    
    if "feature_contributions" in result and result["feature_contributions"]:
        print(f"\n  Top Risk Factors:")
        contributions = result["feature_contributions"]
        sorted_factors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for factor, value in sorted_factors:
            direction = "↑" if value > 0 else "↓"
            print(f"    {direction} {factor}: {value:+.3f}")
    
    return expected == actual


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("   HCT SURVIVAL MODEL - AUTOMATED TESTING")
    print("="*60)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   AI Service: {AI_SERVICE_URL}")
    
    if not check_service():
        print("\n❌ ERROR: AI Service is not running!")
        print("   Please start the service with: docker-compose up -d")
        return
    
    print("\n   Service Status: ✅ Online")
    
    passed = 0
    failed = 0
    
    for case_num, case_info in TEST_CASES.items():
        try:
            result = run_prediction(case_info["data"])
            if print_result(case_num, case_info, result):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Case {case_num} FAILED: {str(e)}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"   SUMMARY: {passed} passed, {failed} failed out of {len(TEST_CASES)}")
    print(f"{'='*60}\n")


def run_single_test(case_num: int):
    """Run a single test case."""
    if case_num not in TEST_CASES:
        print(f"❌ Invalid case number. Available: {list(TEST_CASES.keys())}")
        return
    
    if not check_service():
        print("❌ AI Service is not running!")
        return
    
    case_info = TEST_CASES[case_num]
    result = run_prediction(case_info["data"])
    print_result(case_num, case_info, result)


def run_custom_test():
    """Run with custom data from input."""
    print("\nEnter patient data (press Enter for defaults):")
    
    data = {
        "age_at_hct": int(input("  Age at HCT [45]: ") or 45),
        "donor_age": int(input("  Donor age [35]: ") or 35),
        "karnofsky_score": int(input("  Karnofsky score [80]: ") or 80),
        "comorbidity_score": int(input("  Comorbidity score [2]: ") or 2),
        "disease_status": input("  Disease status [Early/Intermediate/Late] [Intermediate]: ") or "Intermediate",
        "donor_type": input("  Donor type [HLA-identical sibling]: ") or "HLA-identical sibling",
    }
    
    if not check_service():
        print("❌ AI Service is not running!")
        return
    
    result = run_prediction(data)
    
    print(f"\n{'='*60}")
    print("CUSTOM PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"  Risk Category: {result.get('risk_category', 'Unknown')}")
    print(f"  Probability:   {result.get('event_probability', 0)*100:.1f}%")
    print(f"  Confidence:    {result.get('confidence_level', 'N/A')}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--custom":
            run_custom_test()
        elif sys.argv[1] == "--case" and len(sys.argv) > 2:
            run_single_test(int(sys.argv[2]))
        else:
            print("Usage:")
            print("  python test_predictions.py           # Run all tests")
            print("  python test_predictions.py --case N  # Run case N (1-6)")
            print("  python test_predictions.py --custom  # Enter custom data")
    else:
        run_all_tests()
