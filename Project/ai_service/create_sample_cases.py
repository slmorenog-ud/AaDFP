"""
Create diverse sample patient cases for demonstration
"""
import requests
import time

API_URL = "http://localhost:8001"
TOKEN = None

def login():
    global TOKEN
    response = requests.post(f"{API_URL}/auth/login", json={
        "email": "admin@example.com",
        "password": "admin123"
    })
    if response.status_code == 200:
        TOKEN = response.json()['access_token']
        print("✓ Logged in as admin")
        return True
    print(f"Login failed: {response.text}")
    return False

def get_headers():
    return {"Authorization": f"Bearer {TOKEN}"}

# Diverse patient cases - using correct field names from Patient model
SAMPLE_PATIENTS = [
    {
        "name": "María García López",
        "age_at_hct": 28,
        "race_group": "Hispanic",
        "prim_disease_hct": "AML",
        "dri_score": "Low",
        "karnofsky_score": 90,
        "comorbidity_score": 1,
        "donor_related": "MSD",
        "hla_match_c_high": 2,
        "hla_match_drb1_high": 2,
        "conditioning_intensity": "MAC",
        "gvhd_proph": "CNI+MTX",
        "year_hct": 2024
    },
    {
        "name": "Carlos Rodríguez Martínez",
        "age_at_hct": 55,
        "race_group": "Hispanic",
        "prim_disease_hct": "MDS",
        "dri_score": "Intermediate",
        "karnofsky_score": 70,
        "comorbidity_score": 4,
        "donor_related": "MUD",
        "hla_match_c_high": 1,
        "conditioning_intensity": "RIC",
        "diabetes": "Yes",
        "cardiac": "Yes",
        "year_hct": 2024
    },
    {
        "name": "Ana Sofía Pérez",
        "age_at_hct": 8,
        "race_group": "Hispanic",
        "prim_disease_hct": "ALL",
        "dri_score": "Low",
        "karnofsky_score": 100,
        "comorbidity_score": 0,
        "donor_related": "MSD",
        "hla_match_c_high": 2,
        "hla_match_drb1_high": 2,
        "conditioning_intensity": "MAC",
        "year_hct": 2024
    },
    {
        "name": "Roberto Hernández Vega",
        "age_at_hct": 67,
        "race_group": "Hispanic",
        "prim_disease_hct": "MPN",
        "dri_score": "High",
        "karnofsky_score": 60,
        "comorbidity_score": 6,
        "donor_related": "Haplo",
        "conditioning_intensity": "RIC",
        "renal_issue": "Yes",
        "pulm_moderate": "Yes",
        "hepatic_mild": "Yes",
        "year_hct": 2024
    },
    {
        "name": "Isabella Torres Ruiz",
        "age_at_hct": 35,
        "race_group": "Hispanic",
        "prim_disease_hct": "NHL",
        "dri_score": "Low",
        "karnofsky_score": 80,
        "comorbidity_score": 2,
        "donor_related": "Auto",
        "conditioning_intensity": "MAC",
        "prior_tumor": "Yes",
        "year_hct": 2024
    },
    {
        "name": "Diego Morales Castro",
        "age_at_hct": 45,
        "race_group": "Hispanic",
        "prim_disease_hct": "CML",
        "dri_score": "Intermediate",
        "karnofsky_score": 85,
        "comorbidity_score": 3,
        "donor_related": "MUD",
        "hla_match_c_high": 2,
        "hla_match_drb1_high": 2,
        "conditioning_intensity": "MAC",
        "gvhd_proph": "CNI+MTX",
        "obesity": "Yes",
        "year_hct": 2024
    },
    {
        "name": "Valentina Sánchez",
        "age_at_hct": 22,
        "race_group": "Hispanic",
        "prim_disease_hct": "SAA",
        "dri_score": "Low",
        "karnofsky_score": 95,
        "comorbidity_score": 0,
        "donor_related": "MSD",
        "hla_match_c_high": 2,
        "hla_match_drb1_high": 2,
        "conditioning_intensity": "NMA",
        "year_hct": 2024
    },
    {
        "name": "Fernando Gutiérrez Ávila",
        "age_at_hct": 72,
        "race_group": "Hispanic",
        "prim_disease_hct": "AML",
        "dri_score": "Very High",
        "karnofsky_score": 50,
        "comorbidity_score": 8,
        "donor_related": "Haplo",
        "conditioning_intensity": "RIC",
        "cardiac": "Yes",
        "arrhythmia": "Yes",
        "pulm_severe": "Yes",
        "renal_issue": "Yes",
        "diabetes": "Yes",
        "year_hct": 2024
    }
]

def create_patient_and_predict(patient_data):
    try:
        response = requests.post(f"{API_URL}/patients", json=patient_data, headers=get_headers())
        if response.status_code != 200:
            print(f"Error creating {patient_data['name']}: {response.text}")
            return None
        
        patient = response.json()
        patient_id = patient['id']
        
        pred_response = requests.post(f"{API_URL}/predictions", json={"patient_id": patient_id}, headers=get_headers())
        if pred_response.status_code != 200:
            print(f"Error predicting for {patient_data['name']}: {pred_response.text}")
            return None
        
        prediction = pred_response.json()
        
        print(f"✓ {patient_data['name']}")
        print(f"  Age: {patient_data['age_at_hct']}, Karnofsky: {patient_data['karnofsky_score']}, Comorbidity: {patient_data['comorbidity_score']}")
        print(f"  → Risk: {prediction['risk_category']} ({prediction['event_probability']:.1%})")
        print(f"  → Confidence: {prediction.get('confidence_level', 'N/A')}")
        print()
        
        return prediction
        
    except Exception as e:
        print(f"Error with {patient_data['name']}: {e}")
        return None

def main():
    print("=" * 60)
    print("Creating Sample Patient Cases")
    print("=" * 60)
    print()
    
    if not login():
        return
    
    print()
    results = {"Low": 0, "Medium": 0, "High": 0}
    
    for patient in SAMPLE_PATIENTS:
        prediction = create_patient_and_predict(patient)
        if prediction:
            results[prediction['risk_category']] += 1
        time.sleep(0.3)
    
    print("=" * 60)
    print("Summary:")
    print(f"  Low Risk: {results['Low']} patients")
    print(f"  Medium Risk: {results['Medium']} patients")
    print(f"  High Risk: {results['High']} patients")
    print("=" * 60)

if __name__ == "__main__":
    main()
