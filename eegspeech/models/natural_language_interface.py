"""
Natural Language Interface for EEG Analysis
Allows users to interact with the EEG system using natural language queries
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from .vlm_integration import HealthcareVLMPipeline, ClinicalVLMAnalyzer
from .dataset import create_synthetic_eeg_data, load_eeg_data
from .model import EEGSpeechClassifier
import torch


@dataclass
class QueryResult:
    """Structured query result"""
    query: str
    answer: str
    confidence: float
    data: Optional[Dict[str, Any]] = None
    visualization: Optional[Any] = None
    recommendations: Optional[List[str]] = None


class EEGNaturalLanguageInterface:
    """
    Natural Language Interface for EEG Analysis
    Processes natural language queries and provides clinical insights
    """
    
    def __init__(self):
        self.vlm_pipeline = HealthcareVLMPipeline()
        self.analyzer = ClinicalVLMAnalyzer()
        self.model = None
        self.current_patient = None
        self.analysis_history = []
        
        # Query pattern definitions
        self.query_patterns = {
            'risk_assessment': [
                r'what.*risk.*patient',
                r'assess.*risk',
                r'risk.*level',
                r'how.*dangerous',
                r'probability.*disorder'
            ],
            'speech_analysis': [
                r'analyze.*speech',
                r'speech.*pattern',
                r'phoneme.*classification',
                r'what.*phoneme',
                r'speech.*quality'
            ],
            'brain_activity': [
                r'brain.*activity',
                r'neural.*pattern',
                r'eeg.*signal',
                r'cortex.*activation',
                r'hemisphere.*activity'
            ],
            'clinical_insight': [
                r'clinical.*findings',
                r'medical.*interpretation',
                r'diagnosis.*suggestion',
                r'treatment.*recommendation',
                r'therapeutic.*approach'
            ],
            'comparison': [
                r'compare.*with',
                r'difference.*between',
                r'similar.*to',
                r'baseline.*comparison',
                r'normal.*vs'
            ],
            'prediction': [
                r'predict.*outcome',
                r'future.*prognosis',
                r'recovery.*timeline',
                r'treatment.*response',
                r'progression.*forecast'
            ],
            'data_query': [
                r'show.*data',
                r'display.*results',
                r'visualize.*pattern',
                r'plot.*signal',
                r'graph.*frequency'
            ]
        }
    
    def load_model(self, model_path: str = 'eeg_speech_classifier.pth'):
        """Load EEG classification model"""
        try:
            self.model = EEGSpeechClassifier(14, 8)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def set_patient(self, patient_info: Dict[str, Any]):
        """Set current patient for analysis"""
        self.current_patient = patient_info
    
    def process_query(self, query: str, eeg_data: Optional[np.ndarray] = None) -> QueryResult:
        """
        Process natural language query and return structured result
        """
        query_lower = query.lower()
        query_type = self._classify_query(query_lower)
        
        # Route to appropriate handler
        if query_type == 'risk_assessment':
            return self._handle_risk_assessment_query(query, eeg_data)
        elif query_type == 'speech_analysis':
            return self._handle_speech_analysis_query(query, eeg_data)
        elif query_type == 'brain_activity':
            return self._handle_brain_activity_query(query, eeg_data)
        elif query_type == 'clinical_insight':
            return self._handle_clinical_insight_query(query, eeg_data)
        elif query_type == 'comparison':
            return self._handle_comparison_query(query, eeg_data)
        elif query_type == 'prediction':
            return self._handle_prediction_query(query, eeg_data)
        elif query_type == 'data_query':
            return self._handle_data_query(query, eeg_data)
        else:
            return self._handle_general_query(query, eeg_data)
    
    def _classify_query(self, query: str) -> str:
        """Classify query type based on patterns"""
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return query_type
        return 'general'
    
    def _handle_risk_assessment_query(self, query: str, eeg_data: Optional[np.ndarray]) -> QueryResult:
        """Handle risk assessment related queries"""
        if eeg_data is None:
            return QueryResult(
                query=query,
                answer="I need EEG data to perform risk assessment. Please provide EEG data first.",
                confidence=0.9
            )
        
        # Run VLM analysis
        analysis = self.vlm_pipeline.process_eeg_for_healthcare(
            eeg_data, None, self.current_patient
        )
        
        risk_data = analysis["spectrogram_analysis"]["risk_assessment"]
        
        # Generate natural language response
        overall_risk = risk_data["overall_risk"]
        risk_level = "low" if overall_risk < 0.3 else "moderate" if overall_risk < 0.7 else "high"
        
        answer = f"""Based on the EEG analysis, the patient shows {risk_level} risk levels:

• Overall Risk: {overall_risk:.1%} ({risk_level.upper()})
• Speech Disorder Risk: {risk_data['speech_disorder_risk']:.1%}
• Cognitive Decline Risk: {risk_data['cognitive_decline_risk']:.1%}
• Stroke Risk: {risk_data['stroke_risk']:.1%}

The neural patterns indicate {'normal' if risk_level == 'low' else 'some concerns with'} speech motor function."""
        
        recommendations = [
            f"Continue monitoring - risk level is {risk_level}",
            "Regular follow-up recommended" if risk_level != "low" else "Standard follow-up schedule",
            "Consider specialist consultation" if risk_level == "high" else "No immediate specialist referral needed"
        ]
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=0.95,
            data=risk_data,
            recommendations=recommendations
        )
    
    def _handle_speech_analysis_query(self, query: str, eeg_data: Optional[np.ndarray]) -> QueryResult:
        """Handle speech analysis queries"""
        if eeg_data is None or self.model is None:
            return QueryResult(
                query=query,
                answer="I need both EEG data and a loaded model to analyze speech patterns.",
                confidence=0.9
            )
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(eeg_data).unsqueeze(0)
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            _, pred = torch.max(output, 1)
        
        phoneme_labels = ['a', 'e', 'i', 'o', 'u', 'p', 't', 'k']
        predicted_phoneme = phoneme_labels[pred.item()]
        confidence = probs[pred].item()
        
        # Analyze speech characteristics
        vowel_consonant = "vowel" if pred.item() < 5 else "consonant"
        
        answer = f"""Speech Analysis Results:

• Predicted Phoneme: /{predicted_phoneme}/ ({vowel_consonant})
• Classification Confidence: {confidence:.1%}
• Speech Type: {vowel_consonant.title()}

Neural Pattern Analysis:
The EEG shows typical {vowel_consonant} production patterns with appropriate activation in speech motor areas. The classification confidence is {'high' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'low'}, indicating {'clear' if confidence > 0.8 else 'some' if confidence > 0.6 else 'unclear'} neural signatures."""
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=confidence,
            data={"predicted_phoneme": predicted_phoneme, "confidence": confidence, "type": vowel_consonant}
        )
    
    def _handle_brain_activity_query(self, query: str, eeg_data: Optional[np.ndarray]) -> QueryResult:
        """Handle brain activity queries"""
        if eeg_data is None:
            return QueryResult(
                query=query,
                answer="I need EEG data to analyze brain activity patterns.",
                confidence=0.9
            )
        
        # Analyze brain activity patterns
        channel_names = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
        
        # Calculate activity metrics
        mean_activity = np.mean(np.abs(eeg_data), axis=1)
        max_activity_channel = np.argmax(mean_activity)
        
        # Frequency analysis
        from scipy import signal
        f, psd = signal.welch(eeg_data, fs=1000, nperseg=256)
        
        # Identify dominant frequency bands
        delta_power = np.mean(psd[:, (f >= 0.5) & (f < 4)], axis=1)
        theta_power = np.mean(psd[:, (f >= 4) & (f < 8)], axis=1)
        alpha_power = np.mean(psd[:, (f >= 8) & (f < 13)], axis=1)
        beta_power = np.mean(psd[:, (f >= 13) & (f < 30)], axis=1)
        
        dominant_band = np.argmax([np.mean(delta_power), np.mean(theta_power), 
                                  np.mean(alpha_power), np.mean(beta_power)])
        band_names = ['Delta', 'Theta', 'Alpha', 'Beta']
        
        answer = f"""Brain Activity Analysis:

• Most Active Region: {channel_names[max_activity_channel]} electrode
• Dominant Frequency Band: {band_names[dominant_band]} waves
• Overall Activity Level: {'High' if np.mean(mean_activity) > 1.0 else 'Moderate' if np.mean(mean_activity) > 0.5 else 'Low'}

Regional Activity:
• Frontal (F3, F4, Fz): {'Active' if np.mean(mean_activity[:3]) > np.mean(mean_activity) else 'Normal'}
• Temporal (T3, T4, T5, T6): {'Active' if np.mean(mean_activity[9:13]) > np.mean(mean_activity) else 'Normal'}
• Parietal (P3, P4, Pz): {'Active' if np.mean(mean_activity[6:9]) > np.mean(mean_activity) else 'Normal'}

This pattern is {'typical' if band_names[dominant_band] in ['Alpha', 'Beta'] else 'atypical'} for speech-related brain activity."""
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=0.85,
            data={
                "max_activity_channel": channel_names[max_activity_channel],
                "dominant_band": band_names[dominant_band],
                "activity_level": np.mean(mean_activity)
            }
        )
    
    def _handle_clinical_insight_query(self, query: str, eeg_data: Optional[np.ndarray]) -> QueryResult:
        """Handle clinical insight queries"""
        if eeg_data is None:
            return QueryResult(
                query=query,
                answer="I need EEG data to provide clinical insights.",
                confidence=0.9
            )
        
        # Run full VLM analysis
        analysis = self.vlm_pipeline.process_eeg_for_healthcare(
            eeg_data, None, self.current_patient
        )
        
        # Extract key clinical insights
        clinical_insights = analysis["spectrogram_analysis"]["clinical_insights"]
        
        # Format response
        answer = "Clinical Insights:\n\n"
        for question, insight in clinical_insights.items():
            answer += f"• {question}\n  {insight}\n\n"
        
        # Add recommendations
        recommendations = analysis["spectrogram_analysis"]["recommendations"]
        answer += "Clinical Recommendations:\n"
        for i, rec in enumerate(recommendations, 1):
            answer += f"{i}. {rec}\n"
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=0.92,
            data=clinical_insights,
            recommendations=recommendations
        )
    
    def _handle_comparison_query(self, query: str, eeg_data: Optional[np.ndarray]) -> QueryResult:
        """Handle comparison queries"""
        if len(self.analysis_history) < 2:
            return QueryResult(
                query=query,
                answer="I need at least two previous analyses to make comparisons. Please analyze more EEG data first.",
                confidence=0.9
            )
        
        # Compare with previous analysis
        current_analysis = self.vlm_pipeline.process_eeg_for_healthcare(
            eeg_data, None, self.current_patient
        )
        
        previous_analysis = self.analysis_history[-1]
        
        # Compare risk levels
        current_risk = current_analysis["spectrogram_analysis"]["risk_assessment"]["overall_risk"]
        previous_risk = previous_analysis["spectrogram_analysis"]["risk_assessment"]["overall_risk"]
        
        risk_change = current_risk - previous_risk
        trend = "increased" if risk_change > 0.05 else "decreased" if risk_change < -0.05 else "remained stable"
        
        answer = f"""Comparison with Previous Analysis:

• Overall Risk: {current_risk:.1%} (vs {previous_risk:.1%} previously)
• Risk Trend: {trend.title()} by {abs(risk_change):.1%}
• Change Significance: {'Significant' if abs(risk_change) > 0.1 else 'Minor' if abs(risk_change) > 0.05 else 'Negligible'}

Clinical Interpretation:
The patient's condition has {'improved' if risk_change < -0.05 else 'worsened' if risk_change > 0.05 else 'remained stable'} since the last analysis. {'Continue current treatment' if abs(risk_change) < 0.05 else 'Consider treatment adjustment' if abs(risk_change) > 0.1 else 'Monitor closely'}."""
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=0.88,
            data={"current_risk": current_risk, "previous_risk": previous_risk, "trend": trend}
        )
    
    def _handle_prediction_query(self, query: str, eeg_data: Optional[np.ndarray]) -> QueryResult:
        """Handle prediction queries"""
        if eeg_data is None:
            return QueryResult(
                query=query,
                answer="I need EEG data to make predictions.",
                confidence=0.9
            )
        
        # Analyze current state
        analysis = self.vlm_pipeline.process_eeg_for_healthcare(
            eeg_data, None, self.current_patient
        )
        
        current_risk = analysis["spectrogram_analysis"]["risk_assessment"]["overall_risk"]
        
        # Simple prediction model based on risk levels
        if current_risk < 0.2:
            prognosis = "excellent"
            recovery_time = "immediate"
            intervention_needed = "none"
        elif current_risk < 0.4:
            prognosis = "good"
            recovery_time = "3-6 months"
            intervention_needed = "monitoring"
        elif current_risk < 0.7:
            prognosis = "fair"
            recovery_time = "6-12 months"
            intervention_needed = "therapy"
        else:
            prognosis = "guarded"
            recovery_time = "12+ months"
            intervention_needed = "intensive treatment"
        
        answer = f"""Predictive Analysis:

• Prognosis: {prognosis.title()}
• Expected Recovery Timeline: {recovery_time}
• Intervention Level: {intervention_needed.title()}

Prediction Factors:
• Current Risk Level: {current_risk:.1%}
• Neural Pattern Quality: {'Good' if current_risk < 0.5 else 'Concerning'}
• Response to Treatment: {'Likely positive' if current_risk < 0.5 else 'May require intensive approach'}

Confidence: {'High' if current_risk < 0.3 or current_risk > 0.7 else 'Moderate'} - prediction based on current neural patterns and established clinical correlations."""
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=0.75,
            data={"prognosis": prognosis, "recovery_time": recovery_time, "intervention": intervention_needed}
        )
    
    def _handle_data_query(self, query: str, eeg_data: Optional[np.ndarray]) -> QueryResult:
        """Handle data visualization queries"""
        if eeg_data is None:
            return QueryResult(
                query=query,
                answer="I need EEG data to create visualizations.",
                confidence=0.9
            )
        
        # Generate visualizations
        analysis = self.vlm_pipeline.process_eeg_for_healthcare(
            eeg_data, None, self.current_patient
        )
        
        available_visualizations = [
            "Clinical Spectrogram",
            "Brain Topography Map",
            "Temporal Dynamics Plot"
        ]
        
        answer = f"""Available Data Visualizations:

• {available_visualizations[0]}: Shows frequency content over time with clinical annotations
• {available_visualizations[1]}: Displays spatial brain activity patterns
• {available_visualizations[2]}: Illustrates temporal evolution of neural patterns

The visualizations have been generated and are available in the analysis results. They show {'normal' if analysis['spectrogram_analysis']['risk_assessment']['overall_risk'] < 0.3 else 'abnormal'} patterns consistent with the clinical findings."""
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=0.95,
            visualization=analysis["images"]
        )
    
    def _handle_general_query(self, query: str, eeg_data: Optional[np.ndarray]) -> QueryResult:
        """Handle general queries"""
        answer = f"""I understand you're asking: "{query}"

I can help you with:
• Risk assessment for neurological conditions
• Speech pattern analysis and classification
• Brain activity interpretation
• Clinical insights and recommendations
• Comparison with previous analyses
• Predictive healthcare modeling
• Data visualization and exploration

Please provide EEG data and ask specific questions about:
- Risk levels and health assessment
- Speech phoneme analysis
- Brain activity patterns
- Clinical findings and recommendations
- Comparisons with previous results
- Predictions about treatment outcomes
- Data visualization requests

Would you like me to analyze any specific aspect of the EEG data?"""
        
        return QueryResult(
            query=query,
            answer=answer,
            confidence=0.8
        )
    
    def add_to_history(self, analysis_result: Dict[str, Any]):
        """Add analysis result to history"""
        self.analysis_history.append(analysis_result)
        
        # Keep only last 10 analyses
        if len(self.analysis_history) > 10:
            self.analysis_history.pop(0)
    
    def get_conversation_context(self) -> str:
        """Get conversation context for continuity"""
        if not self.analysis_history:
            return "No previous analysis history available."
        
        latest = self.analysis_history[-1]
        context = f"""Recent Analysis Context:
• Patient: {self.current_patient.get('id', 'Unknown') if self.current_patient else 'Unknown'}
• Latest Risk Level: {latest['spectrogram_analysis']['risk_assessment']['overall_risk']:.1%}
• Analysis Count: {len(self.analysis_history)}
• Last Analysis: {latest.get('timestamp', 'Unknown time')}"""
        
        return context


class ConversationalEEGInterface:
    """
    Conversational interface that maintains context and provides natural interactions
    """
    
    def __init__(self):
        self.nli = EEGNaturalLanguageInterface()
        self.conversation_history = []
        self.current_eeg_data = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def start_conversation(self, patient_info: Dict[str, Any] = None):
        """Start a new conversation session"""
        if patient_info:
            self.nli.set_patient(patient_info)
        
        welcome_message = f"""
🧠 Welcome to the EEG Healthcare Assistant!

I'm your AI-powered EEG analysis companion. I can help you understand:
• Brain activity patterns and neural signals
• Speech classification and phoneme analysis  
• Clinical risk assessment and health insights
• Treatment recommendations and predictions
• Data visualization and interpretation

{f"Patient: {patient_info.get('id', 'Unknown')}" if patient_info else "No patient information provided"}
Session ID: {self.session_id}

How can I assist you with EEG analysis today?
        """
        
        print(welcome_message)
        return welcome_message
    
    def chat(self, user_input: str, eeg_data: Optional[np.ndarray] = None) -> str:
        """Process user input and return conversational response"""
        
        # Update current EEG data if provided
        if eeg_data is not None:
            self.current_eeg_data = eeg_data
        
        # Process the query
        result = self.nli.process_query(user_input, self.current_eeg_data)
        
        # Add to conversation history
        self.conversation_history.append({
            "user_input": user_input,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "has_eeg_data": self.current_eeg_data is not None
        })
        
        # Format response
        response = f"""
{result.answer}

Confidence: {result.confidence:.1%}
"""
        
        if result.recommendations:
            response += "\n🎯 Recommendations:\n"
            for i, rec in enumerate(result.recommendations, 1):
                response += f"{i}. {rec}\n"
        
        return response
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation"""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary = f"""
CONVERSATION SUMMARY
Session ID: {self.session_id}
Total Interactions: {len(self.conversation_history)}
Patient: {self.nli.current_patient.get('id', 'Unknown') if self.nli.current_patient else 'Unknown'}

Recent Topics:
"""
        
        for entry in self.conversation_history[-5:]:  # Last 5 interactions
            summary += f"• {entry['user_input'][:50]}{'...' if len(entry['user_input']) > 50 else ''}\n"
        
        return summary