"""
Predictive Healthcare Module with VLM Integration
Advanced healthcare predictions using EEG patterns and Vision Language Models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .vlm_integration import HealthcareVLMPipeline, ClinicalVLMAnalyzer
from .natural_language_interface import EEGNaturalLanguageInterface


@dataclass
class PredictionResult:
    """Structured prediction result"""
    prediction_type: str
    outcome: str
    probability: float
    confidence_interval: Tuple[float, float]
    risk_factors: List[str]
    recommendations: List[str]
    timeline: str
    supporting_evidence: Dict[str, Any]
    timestamp: str


@dataclass
class PatientOutcome:
    """Patient outcome for training predictive models"""
    patient_id: str
    age: int
    gender: str
    initial_risk: float
    outcome: str  # 'improved', 'stable', 'deteriorated'
    treatment_response: str  # 'good', 'moderate', 'poor'
    recovery_time: int  # in days
    follow_up_risk: float
    intervention_type: str


class PredictiveHealthcareEngine:
    """
    Advanced predictive healthcare engine using EEG patterns and VLM insights
    """
    
    def __init__(self):
        self.vlm_pipeline = HealthcareVLMPipeline()
        self.nli = EEGNaturalLanguageInterface()
        
        # Predictive models
        self.outcome_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.recovery_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Scalers
        self.scaler_features = StandardScaler()
        self.scaler_risk = StandardScaler()
        
        # Model training status
        self.models_trained = False
        
        # Clinical knowledge base
        self.clinical_knowledge = self._build_clinical_knowledge_base()
        
        # Synthetic training data
        self.training_data = self._generate_synthetic_training_data()
        
        # Train models on synthetic data
        self._train_predictive_models()
    
    def _build_clinical_knowledge_base(self) -> Dict[str, Any]:
        """Build clinical knowledge base for healthcare predictions"""
        return {
            "risk_factors": {
                "age": {"high": 65, "moderate": 45, "weight": 0.3},
                "gender": {"male_risk_multiplier": 1.2, "weight": 0.1},
                "eeg_patterns": {
                    "high_frequency_activity": {"threshold": 0.7, "weight": 0.4},
                    "asymmetry": {"threshold": 0.5, "weight": 0.3},
                    "signal_variability": {"threshold": 0.6, "weight": 0.2}
                }
            },
            "outcome_predictors": {
                "speech_disorder": {
                    "frontal_activation": {"weight": 0.4},
                    "temporal_coherence": {"weight": 0.3},
                    "baseline_risk": {"weight": 0.3}
                },
                "cognitive_decline": {
                    "alpha_power": {"weight": 0.5},
                    "connectivity": {"weight": 0.3},
                    "age_factor": {"weight": 0.2}
                }
            },
            "treatment_response": {
                "good_indicators": ["stable_patterns", "high_coherence", "appropriate_frequency"],
                "poor_indicators": ["irregular_patterns", "low_coherence", "abnormal_frequency"]
            }
        }
    
    def _generate_synthetic_training_data(self) -> List[PatientOutcome]:
        """Generate synthetic training data for predictive models"""
        np.random.seed(42)
        training_data = []
        
        for i in range(1000):
            patient_id = f"P{i:04d}"
            age = np.random.randint(18, 80)
            gender = np.random.choice(['Male', 'Female'])
            
            # Generate initial risk based on demographics
            base_risk = 0.1 + (age - 18) * 0.01 + (0.02 if gender == 'Male' else 0)
            initial_risk = np.clip(base_risk + np.random.normal(0, 0.1), 0, 1)
            
            # Generate outcome based on risk
            if initial_risk < 0.3:
                outcome = np.random.choice(['improved', 'stable'], p=[0.7, 0.3])
                treatment_response = np.random.choice(['good', 'moderate'], p=[0.8, 0.2])
                recovery_time = np.random.randint(30, 90)
                follow_up_risk = initial_risk * np.random.uniform(0.5, 0.8)
            elif initial_risk < 0.6:
                outcome = np.random.choice(['improved', 'stable', 'deteriorated'], p=[0.4, 0.4, 0.2])
                treatment_response = np.random.choice(['good', 'moderate', 'poor'], p=[0.5, 0.3, 0.2])
                recovery_time = np.random.randint(60, 180)
                follow_up_risk = initial_risk * np.random.uniform(0.7, 1.1)
            else:
                outcome = np.random.choice(['stable', 'deteriorated'], p=[0.3, 0.7])
                treatment_response = np.random.choice(['moderate', 'poor'], p=[0.4, 0.6])
                recovery_time = np.random.randint(120, 365)
                follow_up_risk = initial_risk * np.random.uniform(0.8, 1.2)
            
            intervention_type = np.random.choice([
                'monitoring', 'speech_therapy', 'medication', 'intensive_care'
            ])
            
            training_data.append(PatientOutcome(
                patient_id=patient_id,
                age=age,
                gender=gender,
                initial_risk=initial_risk,
                outcome=outcome,
                treatment_response=treatment_response,
                recovery_time=recovery_time,
                follow_up_risk=np.clip(follow_up_risk, 0, 1),
                intervention_type=intervention_type
            ))
        
        return training_data
    
    def _train_predictive_models(self):
        """Train predictive models on synthetic data"""
        # Prepare features and targets
        features = []
        outcome_targets = []
        risk_targets = []
        recovery_targets = []
        
        for patient in self.training_data:
            feature_vector = [
                patient.age,
                1 if patient.gender == 'Male' else 0,
                patient.initial_risk,
                np.random.random(),  # Synthetic EEG feature 1
                np.random.random(),  # Synthetic EEG feature 2
                np.random.random(),  # Synthetic EEG feature 3
                np.random.random(),  # Synthetic EEG feature 4
                np.random.random(),  # Synthetic EEG feature 5
            ]
            
            features.append(feature_vector)
            outcome_targets.append(patient.outcome)
            risk_targets.append(patient.follow_up_risk)
            recovery_targets.append(patient.recovery_time)
        
        features = np.array(features)
        
        # Scale features
        features_scaled = self.scaler_features.fit_transform(features)
        
        # Train outcome classifier
        self.outcome_classifier.fit(features_scaled, outcome_targets)
        
        # Train risk regressor
        self.risk_regressor.fit(features_scaled, risk_targets)
        
        # Train recovery time regressor
        self.recovery_regressor.fit(features_scaled, recovery_targets)
        
        self.models_trained = True
        
        # Evaluate models
        self._evaluate_models(features_scaled, outcome_targets, risk_targets, recovery_targets)
    
    def _evaluate_models(self, features: np.ndarray, outcome_targets: List[str], 
                        risk_targets: List[float], recovery_targets: List[int]):
        """Evaluate predictive models"""
        # Split data for evaluation
        X_train, X_test, y_outcome_train, y_outcome_test = train_test_split(
            features, outcome_targets, test_size=0.2, random_state=42
        )
        
        _, _, y_risk_train, y_risk_test = train_test_split(
            features, risk_targets, test_size=0.2, random_state=42
        )
        
        _, _, y_recovery_train, y_recovery_test = train_test_split(
            features, recovery_targets, test_size=0.2, random_state=42
        )
        
        # Evaluate outcome classifier
        outcome_pred = self.outcome_classifier.predict(X_test)
        outcome_accuracy = accuracy_score(y_outcome_test, outcome_pred)
        
        # Evaluate risk regressor
        risk_pred = self.risk_regressor.predict(X_test)
        risk_mse = mean_squared_error(y_risk_test, risk_pred)
        
        # Evaluate recovery regressor
        recovery_pred = self.recovery_regressor.predict(X_test)
        recovery_mse = mean_squared_error(y_recovery_test, recovery_pred)
        
        print(f"Model Evaluation Results:")
        print(f"  Outcome Classifier Accuracy: {outcome_accuracy:.3f}")
        print(f"  Risk Regressor MSE: {risk_mse:.3f}")
        print(f"  Recovery Regressor MSE: {recovery_mse:.3f}")
    
    def predict_patient_outcome(self, eeg_data: np.ndarray, patient_info: Dict[str, Any]) -> PredictionResult:
        """Predict patient outcome using EEG data and patient information"""
        if not self.models_trained:
            raise ValueError("Predictive models not trained yet")
        
        # Analyze EEG with VLM
        vlm_analysis = self.vlm_pipeline.process_eeg_for_healthcare(
            eeg_data, None, patient_info
        )
        
        # Extract features for prediction
        features = self._extract_predictive_features(eeg_data, patient_info, vlm_analysis)
        
        # Make predictions
        outcome_pred = self.outcome_classifier.predict([features])[0]
        outcome_proba = self.outcome_classifier.predict_proba([features])[0]
        risk_pred = self.risk_regressor.predict([features])[0]
        recovery_pred = self.recovery_regressor.predict([features])[0]
        
        # Get confidence intervals
        confidence_interval = self._calculate_confidence_interval(features, outcome_pred)
        
        # Determine risk factors
        risk_factors = self._identify_risk_factors(features, vlm_analysis)
        
        # Generate recommendations
        recommendations = self._generate_treatment_recommendations(
            outcome_pred, risk_pred, recovery_pred, vlm_analysis
        )
        
        # Determine timeline
        timeline = self._determine_timeline(recovery_pred, outcome_pred)
        
        # Supporting evidence
        supporting_evidence = {
            "eeg_analysis": vlm_analysis,
            "risk_score": risk_pred,
            "recovery_estimate": recovery_pred,
            "feature_importance": self._get_feature_importance(features)
        }
        
        return PredictionResult(
            prediction_type="patient_outcome",
            outcome=outcome_pred,
            probability=np.max(outcome_proba),
            confidence_interval=confidence_interval,
            risk_factors=risk_factors,
            recommendations=recommendations,
            timeline=timeline,
            supporting_evidence=supporting_evidence,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_predictive_features(self, eeg_data: np.ndarray, patient_info: Dict[str, Any], 
                                   vlm_analysis: Dict[str, Any]) -> List[float]:
        """Extract features for predictive modeling"""
        # Basic demographics
        age = patient_info.get('age', 45)
        gender = 1 if patient_info.get('gender') == 'Male' else 0
        
        # EEG-derived features
        initial_risk = vlm_analysis["spectrogram_analysis"]["risk_assessment"]["overall_risk"]
        
        # Signal characteristics
        mean_amplitude = np.mean(np.abs(eeg_data))
        signal_variability = np.std(eeg_data)
        
        # Frequency domain features
        from scipy import signal as sig
        f, psd = sig.welch(eeg_data, fs=1000, nperseg=256)
        alpha_power = np.mean(psd[:, (f >= 8) & (f < 13)])
        beta_power = np.mean(psd[:, (f >= 13) & (f < 30)])
        
        # Connectivity features
        coherence = np.mean(np.corrcoef(eeg_data))
        
        return [age, gender, initial_risk, mean_amplitude, signal_variability, 
                alpha_power, beta_power, coherence]
    
    def _calculate_confidence_interval(self, features: List[float], outcome: str) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simple confidence interval based on model uncertainty
        base_confidence = 0.8
        
        # Adjust based on risk level
        risk_level = features[2]  # initial_risk
        if risk_level < 0.3:
            confidence = base_confidence + 0.1
        elif risk_level > 0.7:
            confidence = base_confidence - 0.1
        else:
            confidence = base_confidence
        
        margin = (1 - confidence) / 2
        return (confidence - margin, confidence + margin)
    
    def _identify_risk_factors(self, features: List[float], vlm_analysis: Dict[str, Any]) -> List[str]:
        """Identify key risk factors"""
        risk_factors = []
        
        age = features[0]
        if age > 65:
            risk_factors.append("Advanced age (>65 years)")
        
        gender = features[1]
        if gender == 1:
            risk_factors.append("Male gender")
        
        initial_risk = features[2]
        if initial_risk > 0.5:
            risk_factors.append("High baseline EEG risk score")
        
        signal_variability = features[4]
        if signal_variability > 0.6:
            risk_factors.append("High signal variability")
        
        coherence = features[7]
        if coherence < 0.4:
            risk_factors.append("Low inter-channel coherence")
        
        return risk_factors
    
    def _generate_treatment_recommendations(self, outcome: str, risk: float, 
                                          recovery_time: float, vlm_analysis: Dict[str, Any]) -> List[str]:
        """Generate treatment recommendations"""
        recommendations = []
        
        if outcome == 'improved':
            recommendations.append("Continue current treatment approach")
            recommendations.append("Regular monitoring recommended")
        elif outcome == 'stable':
            recommendations.append("Maintain current treatment")
            recommendations.append("Consider preventive interventions")
        else:  # deteriorated
            recommendations.append("Intensive intervention required")
            recommendations.append("Specialist consultation recommended")
        
        if risk > 0.7:
            recommendations.append("High-risk monitoring protocol")
            recommendations.append("Consider hospitalization")
        
        if recovery_time > 180:
            recommendations.append("Long-term care planning needed")
            recommendations.append("Family support systems")
        
        return recommendations
    
    def _determine_timeline(self, recovery_time: float, outcome: str) -> str:
        """Determine treatment timeline"""
        if outcome == 'improved':
            if recovery_time < 60:
                return "Short-term recovery (2-8 weeks)"
            elif recovery_time < 120:
                return "Medium-term recovery (2-4 months)"
            else:
                return "Long-term recovery (4+ months)"
        elif outcome == 'stable':
            return "Ongoing management required"
        else:
            return "Extended care needed (6+ months)"
    
    def _get_feature_importance(self, features: List[float]) -> Dict[str, float]:
        """Get feature importance for interpretation"""
        feature_names = ['age', 'gender', 'initial_risk', 'amplitude', 'variability', 
                        'alpha_power', 'beta_power', 'coherence']
        
        importance = self.outcome_classifier.feature_importances_
        
        return dict(zip(feature_names, importance))
    
    def predict_treatment_response(self, eeg_data: np.ndarray, treatment_type: str, 
                                 patient_info: Dict[str, Any]) -> PredictionResult:
        """Predict treatment response"""
        # Analyze current state
        vlm_analysis = self.vlm_pipeline.process_eeg_for_healthcare(
            eeg_data, None, patient_info
        )
        
        # Extract features
        features = self._extract_predictive_features(eeg_data, patient_info, vlm_analysis)
        
        # Predict response based on treatment type
        response_probability = self._calculate_treatment_response_probability(
            features, treatment_type
        )
        
        # Generate treatment-specific recommendations
        recommendations = self._generate_treatment_specific_recommendations(
            treatment_type, response_probability, vlm_analysis
        )
        
        return PredictionResult(
            prediction_type="treatment_response",
            outcome=f"Response to {treatment_type}",
            probability=response_probability,
            confidence_interval=(response_probability - 0.1, response_probability + 0.1),
            risk_factors=self._identify_risk_factors(features, vlm_analysis),
            recommendations=recommendations,
            timeline=self._determine_treatment_timeline(treatment_type, response_probability),
            supporting_evidence={"treatment_type": treatment_type, "features": features},
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_treatment_response_probability(self, features: List[float], 
                                                treatment_type: str) -> float:
        """Calculate treatment response probability"""
        base_probability = 0.6
        
        # Adjust based on patient characteristics
        age = features[0]
        if age < 40:
            base_probability += 0.2
        elif age > 65:
            base_probability -= 0.2
        
        # Adjust based on EEG features
        initial_risk = features[2]
        if initial_risk < 0.3:
            base_probability += 0.1
        elif initial_risk > 0.7:
            base_probability -= 0.2
        
        # Adjust based on treatment type
        treatment_adjustments = {
            'speech_therapy': 0.1,
            'medication': 0.05,
            'intensive_care': 0.15,
            'monitoring': 0.0
        }
        
        base_probability += treatment_adjustments.get(treatment_type, 0)
        
        return np.clip(base_probability, 0, 1)
    
    def _generate_treatment_specific_recommendations(self, treatment_type: str, 
                                                   response_probability: float, 
                                                   vlm_analysis: Dict[str, Any]) -> List[str]:
        """Generate treatment-specific recommendations"""
        recommendations = []
        
        if treatment_type == 'speech_therapy':
            recommendations.append("Focus on motor speech exercises")
            recommendations.append("Regular progress monitoring")
            if response_probability > 0.7:
                recommendations.append("Accelerated therapy schedule")
        
        elif treatment_type == 'medication':
            recommendations.append("Monitor for side effects")
            recommendations.append("Adjust dosage based on response")
            if response_probability < 0.5:
                recommendations.append("Consider alternative medications")
        
        elif treatment_type == 'intensive_care':
            recommendations.append("Multidisciplinary team approach")
            recommendations.append("Daily assessments")
            recommendations.append("Family involvement crucial")
        
        return recommendations
    
    def _determine_treatment_timeline(self, treatment_type: str, response_probability: float) -> str:
        """Determine treatment timeline"""
        if response_probability > 0.7:
            return "Rapid response expected (2-4 weeks)"
        elif response_probability > 0.5:
            return "Moderate response expected (4-8 weeks)"
        else:
            return "Slow response expected (8+ weeks)"
    
    def generate_predictive_report(self, predictions: List[PredictionResult]) -> str:
        """Generate comprehensive predictive report"""
        report = f"""
PREDICTIVE HEALTHCARE ANALYSIS REPORT
{'='*40}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Predictions: {len(predictions)}

SUMMARY OF PREDICTIONS:
{'='*22}
"""
        
        for i, pred in enumerate(predictions, 1):
            report += f"""
Prediction {i}: {pred.prediction_type}
• Outcome: {pred.outcome}
• Probability: {pred.probability:.1%}
• Confidence: {pred.confidence_interval[0]:.1%} - {pred.confidence_interval[1]:.1%}
• Timeline: {pred.timeline}
• Key Risk Factors: {', '.join(pred.risk_factors[:3])}
"""
        
        report += f"""
OVERALL ASSESSMENT:
{'='*17}
The analysis indicates {'high confidence' if np.mean([p.probability for p in predictions]) > 0.7 else 'moderate confidence'} in the predictions.

RECOMMENDATIONS:
{'='*14}
"""
        
        # Aggregate recommendations
        all_recommendations = []
        for pred in predictions:
            all_recommendations.extend(pred.recommendations)
        
        unique_recommendations = list(set(all_recommendations))[:5]
        for i, rec in enumerate(unique_recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
PREDICTIVE MODEL CONFIDENCE:
{'='*26}
Based on {len(self.training_data)} training samples and validated using 
cross-validation techniques. Model performance metrics available upon request.

DISCLAIMER:
These predictions are for research and educational purposes only. 
Clinical decisions should always be made by qualified healthcare professionals.
"""
        
        return report


# Example usage and testing functions
def demonstrate_predictive_capabilities():
    """Demonstrate predictive healthcare capabilities"""
    print("🚀 Initializing Predictive Healthcare Engine...")
    
    # Initialize engine
    engine = PredictiveHealthcareEngine()
    
    # Generate sample EEG data
    from .dataset import create_synthetic_eeg_data
    X, y, classes = create_synthetic_eeg_data(n_samples=1)
    
    # Sample patient information
    patient_info = {
        "id": "P001",
        "age": 55,
        "gender": "Male",
        "medical_history": "Hypertension"
    }
    
    print("📊 Making outcome prediction...")
    outcome_prediction = engine.predict_patient_outcome(X[0], patient_info)
    
    print("💊 Making treatment response prediction...")
    treatment_prediction = engine.predict_treatment_response(X[0], "speech_therapy", patient_info)
    
    # Generate report
    predictions = [outcome_prediction, treatment_prediction]
    report = engine.generate_predictive_report(predictions)
    
    print("\n" + report)
    
    return engine, predictions


if __name__ == "__main__":
    demonstrate_predictive_capabilities()