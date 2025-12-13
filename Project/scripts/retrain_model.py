#!/usr/bin/env python3
"""
Model Retraining Script
========================
Retrains the HCT prediction model with:
- 45 features (instead of 25)
- Forced inclusion of all comorbidities
- 5-fold stratified cross-validation
- Enhanced clinical features

Usage:
    python retrain_model.py
"""

import requests
import json
import time
import sys

AI_SERVICE_URL = "http://localhost:8000"


def check_service():
    """Check if AI service is running."""
    try:
        resp = requests.get(f"{AI_SERVICE_URL}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


def get_current_model_info():
    """Get current model information."""
    resp = requests.get(f"{AI_SERVICE_URL}/model-info")
    return resp.json()


def train_new_model():
    """Train a new model with enhanced features."""
    print("\n" + "="*60)
    print("   HCT MODEL RETRAINING")
    print("="*60)
    
    if not check_service():
        print("\n❌ AI Service is not running!")
        print("   Start with: docker-compose up -d")
        return False
    
    print("\n✅ AI Service is online")
    
    # Get current model info
    current = get_current_model_info()
    print(f"\n📊 Current Model:")
    print(f"   - Features: {current.get('n_features', 'Unknown')}")
    print(f"   - Type: {current.get('model_type', 'Unknown')}")
    print(f"   - Trained: {current.get('training_date', 'Unknown')}")
    
    print("\n🔄 Starting retraining with enhanced configuration...")
    print("   - Features: 45 (including all comorbidities)")
    print("   - Model: GBM (Gradient Boosting)")
    print("   - Validation: 5-fold Stratified CV")
    print("\n   This may take 5-10 minutes...\n")
    
    try:
        # Send training request
        resp = requests.post(
            f"{AI_SERVICE_URL}/train",
            json={
                "data_path": "/app/data/raw/raw/train.csv",
                "model_type": "gbm",
                "n_features": 45
            },
            timeout=600  # 10 minute timeout
        )
        
        if resp.status_code == 200:
            result = resp.json()
            print("\n" + "="*60)
            print("   ✅ TRAINING COMPLETE!")
            print("="*60)
            
            # Parse results
            if 'results' in result:
                stages = result['results'].get('stages', {})
                
                # Preprocessing info
                preproc = stages.get('preprocessing', {})
                print(f"\n📋 Data:")
                print(f"   - Rows: {preproc.get('total_rows', 'N/A')}")
                print(f"   - Columns: {preproc.get('total_columns', 'N/A')}")
                
                # Feature selection
                features = stages.get('feature_selection', {})
                print(f"\n🔍 Features Selected: {features.get('n_features', 'N/A')}")
                if 'clinical_features' in features:
                    print(f"   - Clinical: {len(features.get('clinical_features', []))}")
                if 'comorbidity_features_included' in features:
                    print(f"   - Comorbidities: {len(features.get('comorbidity_features_included', []))}")
                
                # Modeling metrics
                modeling = stages.get('modeling', {})
                cv = modeling.get('cv_metrics', {})
                print(f"\n📈 Cross-Validation Results (5-fold):")
                print(f"   - Mean AUC: {cv.get('mean_auc', 'N/A'):.4f}" if isinstance(cv.get('mean_auc'), float) else f"   - Mean AUC: {cv.get('mean_auc', 'N/A')}")
                print(f"   - Std AUC: {cv.get('std_auc', 'N/A'):.4f}" if isinstance(cv.get('std_auc'), float) else f"   - Std AUC: {cv.get('std_auc', 'N/A')}")
                
                # Fairness
                fairness = stages.get('fairness', {})
                print(f"\n⚖️ Fairness Metrics:")
                print(f"   - Passed: {'✅' if fairness.get('fairness_passed') else '❌'}")
                
            return True
        else:
            print(f"\n❌ Training failed: {resp.status_code}")
            print(resp.text)
            return False
            
    except requests.exceptions.Timeout:
        print("\n⏰ Training timed out (>10 min)")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def run_tests():
    """Run test predictions after training."""
    print("\n" + "="*60)
    print("   RUNNING VALIDATION TESTS")
    print("="*60)
    
    # Import and run tests
    try:
        import test_predictions
        test_predictions.run_all_tests()
    except ImportError:
        print("   (Run test_predictions.py separately)")


if __name__ == "__main__":
    success = train_new_model()
    
    if success:
        print("\n🔄 Running validation tests...")
        time.sleep(2)
        run_tests()
    
    print("\n" + "="*60)
    print("   Done!")
    print("="*60 + "\n")
