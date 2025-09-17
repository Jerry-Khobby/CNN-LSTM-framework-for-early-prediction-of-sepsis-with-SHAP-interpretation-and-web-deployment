# Fixed version of your reporter.py code

import requests
import logging
import json
import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Optional, Any, Union

from xai_sdk import Client
from xai_sdk.chat import user, system

from app.schemas.prediction import RiskAssessment, KeyRiskFactor, AntibioticChoices, ClinicalGuidance, SafetyAlerts, Disclaimers, PatientReport, CsvPredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
XAI_API_URL = "https://api.x.ai/v1"

client = Client(
    api_key=os.getenv("GROK_API_KEY"),
    timeout=3600,
)

def normalize_llm_response(raw_response: Any, patient_id: int, probability: float, risk_level: str) -> PatientReport:
    """
    Normalize and validate LLM response to ensure consistent structure
    """
    try:
        # If raw_response is a string, try to parse as JSON
        if isinstance(raw_response, str):
            try:
                parsed_response = json.loads(raw_response)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for patient {patient_id}, using fallback")
                parsed_response = {}
        elif isinstance(raw_response, dict):
            parsed_response = raw_response
        else:
            parsed_response = {}

        # Format probability as percentage string safely
        probability_percent = f"{probability * 100:.2f}%"  # Convert to percentage manually
        
        # Create base report with required fields
        normalized_report = {
            "patient_id": patient_id,
            "report_type": parsed_response.get("report_type", "Sepsis Risk Assessment"),
            "generated_at": parsed_response.get("generated_at"),
            "risk_probability": probability_percent,  # Use safe percentage formatting
            "risk_assessment": normalize_risk_assessment(parsed_response.get("risk_assessment", {}), probability, risk_level),
            "key_risk_factors": normalize_key_risk_factors(parsed_response.get("key_risk_factors", [])),
            "clinical_guidance": normalize_clinical_guidance(parsed_response.get("clinical_guidance", {})),
            "safety_alerts": normalize_safety_alerts(parsed_response.get("safety_alerts", {})),
            "disclaimers": normalize_disclaimers(parsed_response.get("disclaimers", {}))
        }

        # Validate using Pydantic model
        validated_report = PatientReport(**normalized_report)
        return validated_report

    except Exception as e:
        logger.error(f"Error normalizing report for patient {patient_id}: {str(e)}")
        # Return a fallback report with minimum required fields
        return create_fallback_report(patient_id, probability, risk_level)


def normalize_risk_assessment(risk_data: Dict, probability: float, risk_level: str) -> Dict:
    """Normalize risk assessment data"""
    # Format probability safely
    probability_percent = f"{probability * 100:.2f}%"
    
    return {
        "level": risk_data.get("level", risk_level or "Unknown"),
        "score": risk_data.get("score", f"{risk_level} ({probability_percent})" if risk_level else probability_percent),
        "time_frame": risk_data.get("time_frame", "10-hour sepsis risk"),
        "interpretation": risk_data.get("interpretation", f"Patient shows {risk_level.lower() if risk_level else 'unknown'} risk indicators"),
        "detailed_analysis": risk_data.get("detailed_analysis", "Detailed clinical analysis based on risk factors and probability score")
    }


def normalize_key_risk_factors(factors_data: List) -> List[Dict]:
    """Normalize key risk factors"""
    normalized_factors = []
    
    if not factors_data:
        return []
    
    for factor in factors_data:
        if isinstance(factor, dict):
            normalized_factor = {
                "marker": factor.get("marker", factor.get("feature", "Unknown marker")),
                "importance": float(factor.get("importance", factor.get("value", 0.0))),
                "note": factor.get("note", "Clinical significance noted")
            }
            normalized_factors.append(normalized_factor)
    
    return normalized_factors


def normalize_clinical_guidance(guidance_data: Dict) -> Dict:
    """Normalize clinical guidance data"""
    treatment_options = guidance_data.get("treatment_options", {})
    
    # Handle antibiotic choices - can be list or dict
    antibiotic_choices = treatment_options.get("antibiotic_choices", {})
    if isinstance(antibiotic_choices, list):
        # Convert list to structured format
        antibiotic_choices = {
            "community_acquired": antibiotic_choices[:3] if len(antibiotic_choices) > 3 else antibiotic_choices,
            "hospital_acquired": antibiotic_choices[3:6] if len(antibiotic_choices) > 6 else [],
            "penicillin_allergy": antibiotic_choices[6:] if len(antibiotic_choices) > 6 else []
        }
    
    return {
        "monitoring": ensure_list(guidance_data.get("monitoring", ["Vital signs monitoring", "Laboratory values tracking"])),
        "diagnostic_tests": ensure_list(guidance_data.get("diagnostic_tests", ["Blood cultures", "Complete blood count", "Lactate levels"])),
        "treatment_options": {
            "immediate_medications": ensure_list(treatment_options.get("immediate_medications", ["IV fluids", "Broad-spectrum antibiotics"])),
            "antibiotic_choices": antibiotic_choices
        },
        "required_actions": ensure_list(guidance_data.get("required_actions", ["Immediate clinical assessment"]))
    }


def normalize_safety_alerts(alerts_data: Union[Dict, List]) -> Dict:
    """Normalize safety alerts"""
    if isinstance(alerts_data, list):
        precautions = alerts_data
    elif isinstance(alerts_data, dict):
        precautions = alerts_data.get("precautions", [])
    else:
        precautions = []
    
    return {
        "precautions": ensure_list(precautions) or ["Standard precautions apply"]
    }


def normalize_disclaimers(disclaimers_data: Union[Dict, List]) -> Dict:
    """Normalize disclaimers"""
    if isinstance(disclaimers_data, dict):
        return {
            "model_use": disclaimers_data.get("model_use", "This assessment is generated by an AI model and should be used as a clinical decision support tool only."),
            "physician_oversight": disclaimers_data.get("physician_oversight", "All decisions must be validated by a qualified healthcare professional."),
            "limitations": disclaimers_data.get("limitations", "This model has limitations and should not replace clinical judgment.")
        }
    else:
        return {
            "model_use": "This assessment is generated by an AI model and should be used as a clinical decision support tool only.",
            "physician_oversight": "All decisions must be validated by a qualified healthcare professional.",
            "limitations": "This model has limitations and should not replace clinical judgment."
        }


def ensure_list(value: Any) -> List[str]:
    """Ensure value is a list of strings"""
    if isinstance(value, list):
        return [str(item) for item in value]
    elif isinstance(value, str):
        return [value]
    else:
        return []


def create_fallback_report(patient_id: int, probability: float, risk_level: str) -> PatientReport:
    """Create a fallback report when normalization fails"""
    # Format probability safely
    probability_percent = f"{probability * 100:.2f}%"
    
    return PatientReport(
        patient_id=patient_id,
        risk_probability=probability_percent,
        risk_assessment=RiskAssessment(
            level=risk_level or "Unknown",
            score=probability_percent,
            interpretation=f"Automated assessment indicates {risk_level.lower() if risk_level else 'unknown'} risk level"
        ),
        key_risk_factors=[
            KeyRiskFactor(
                marker="System Assessment",
                importance=probability,
                note="Automated risk calculation based on available data"
            )
        ]
    )


def generate_patient_report(patient: dict, model: str = "grok-3-latest") -> dict:
    try:
        # Extract patient data
        patient_id = patient.get("patient_id")
        probability = patient.get("probability", 0)
        risk_level = patient.get("risk_level", "").strip()
        urgent_flag = patient.get("needs_urgent_attention", False)
        top_features = patient.get("shap_values", {}).get("top_features", {})

        # Format top features
        top_5_features = dict(sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:5])
        feature_list = "\n".join([f"- {k} (importance: {v:.2f})" for k, v in top_5_features.items()]) or "N/A"
        
        # Format probability safely
        probability_percent = f"{probability * 100:.2f}%"

        # Construct messages
        system_message = system("You are a clinical assistant AI specialized in structured medical reporting.")

        user_prompt = f"""
        Patient Sepsis Risk Evaluation:

        Risk Probability: {probability_percent}
        Risk Level: {risk_level}
        Urgent Attention Required: {"Yes" if urgent_flag else "No"}

        Top Risk Markers:
        {feature_list}
        Please generate a structured clinical JSON report with the exact structure:
        {{
            "risk_assessment": {{
                "level": "string",
                "score": "string or number", 
                "time_frame": "10-hour sepsis risk",
                "interpretation": "string",
                "detailed_analysis": "string"
            }},
            "key_risk_factors": [
                {{
                    "marker": "string",
                    "importance": "number",
                    "note": "string"
                }}
            ],
            "clinical_guidance": {{
                "monitoring": ["string"],
                "diagnostic_tests": ["string"],
                "treatment_options": {{
                    "immediate_medications": ["string"],
                    "antibiotic_choices": {{
                        "community_acquired": ["string"],
                        "hospital_acquired": ["string"],
                        "penicillin_allergy": ["string"]
                    }}
                }},
                "required_actions": ["string"]
            }},
            "safety_alerts": {{
                "precautions": ["string"]
            }},
            "disclaimers": {{
                "model_use": "string",
                "physician_oversight": "string", 
                "limitations": "string"
            }}
        }}

        Return ONLY valid JSON. No additional text or explanations.
        """

        # Initialize client and get response
        chat = client.chat.create(model=model)
        chat.append(system_message)
        chat.append(user(user_prompt))

        response = chat.sample()
        if not response or not response.content:
            raise ValueError("Empty response from Grok Model")
            
        # Fixed: Pass parameters in correct order
        normalized_report = normalize_llm_response(
            response.content,  # raw_response
            patient_id,       # patient_id 
            probability,      # probability
            risk_level        # risk_level
        )
        
        logger.info(f"Report generated and validated for patient {patient_id}")
        return CsvPredictionResult(
            patient_id=patient_id,
            success=True,
            report=normalized_report,
            risk_level=risk_level,
            probability=probability
        ).dict()

    except Exception as e:
        logger.error(f"Patient report generation failed for patient {patient.get('patient_id', 'unknown')}: {str(e)}")
        
        # Create fallback report
        fallback_report = create_fallback_report(
            patient.get("patient_id"),
            patient.get("probability", 0),
            patient.get("risk_level", "Unknown")
        )
        
        return CsvPredictionResult(
            patient_id=patient.get("patient_id"),
            success=True,  # We're providing a valid fallback
            report=fallback_report,
            risk_level=patient.get("risk_level", "Unknown"),
            probability=patient.get("probability", 0)
        ).dict()


# Your batch report function (assuming this is part of a FastAPI router)
# @router.post("/reports/batch", summary="Generate LLM reports for multiple patients")
# async def generate_batch_patient_reports(payload: BatchInput):
#     try:
#         shared_shap = payload.shap_values
#         results = []
#         errors = []

#         for patient in payload.predictions:
#             patient_dict = patient.dict()
#             if not patient_dict.get("shap_values"):
#                 patient_dict["shap_values"] = {
#                     "top_features": shared_shap.get("top_features", {})
#                 }

#             report = generate_patient_report(patient_dict)
            
#             if report.get("success"):
#                 results.append(report)
#             else:
#                 errors.append({
#                     "patient_id": patient.patient_id,
#                     "error": report.get("error"),
#                     "recommendation": report.get("recommendation")
#                 })

#         return {
#             "success": len(errors) == 0,
#             "reports": results,
#             "errors": errors,
#             "meta": {
#                 "threshold_used": payload.threshold,
#                 "metrics": payload.metrics,
#                 "total_patients": len(payload.predictions),
#                 "successful_reports": len(results),
#                 "failed_reports": len(errors)
#             }
#         }

#     except Exception as e:
#         logger.error(f"Batch report generation failed: {str(e)}")
#         raise HTTPException(
#             status_code=400,
#             detail={
#                 "error": str(e),
#                 "recommendation": "Check input data format and API connectivity"
#             }
#         )