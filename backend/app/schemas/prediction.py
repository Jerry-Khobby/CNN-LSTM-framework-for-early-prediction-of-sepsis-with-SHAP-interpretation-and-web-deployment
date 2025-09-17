"""
This file defines the Pydantic models for prediction requests and responses in the sepsis prediction API. It includes models for prediction requests and responses, including predictions, probabilities, explanations, and reports.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional,Any,Union

class PredictionRequest(BaseModel):
    Unit1: float
    Gender: float
    HospAdmTime: float
    Age: float
    DBP: float
    Temp: float
    Glucose: float
    Potassium: float
    Hct: float
    FiO2: float
    Hgb: float
    pH: float
    BUN: float
    WBC: float
    Magnesium: float
    Creatinine: float
    Platelets: float
    Calcium: float
    PaCO2: float
    BaseExcess: float
    Chloride: float
    HCO3: float
    Phosphate: float
    EtCO2: float
    SaO2: float
    PTT: float
    Lactate: float
    AST: float
    Alkalinephos: float
    Bilirubin_total: float
    TroponinI: float
    Fibrinogen: float
    Bilirubin_direct: float
    X_cat: List[float]  # Categorical/static features
    X_cont: List[float]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    explanations: Dict[str, float]
    report: str 
    
    
class ReportRequest(BaseModel):
    email: str =None 
    
    

class PredictionRequest(BaseModel):
    X_cat: List[List[Any]]
    X_cont: List[List[float]]
    



from typing import List
from pydantic import BaseModel, Field, validator
import numpy as np
class ManualPredictionRequest(BaseModel):
    # Time-series features (exactly 10 values)
    HR: List[float] = Field(..., min_items=10, max_items=10)
    MAP: List[float] = Field(..., min_items=10, max_items=10)
    O2Sat: List[float] = Field(..., min_items=10, max_items=10)
    SBP: List[float] = Field(..., min_items=10, max_items=10)
    Resp: List[float] = Field(..., min_items=10, max_items=10)
    
    # Categorical features with validation
    Unit1: float = Field(..., ge=0, le=1)
    Gender: float = Field(..., ge=0, le=1)
    HospAdmTime: float = Field(..., ge=0)
    Age: float = Field(..., gt=0, le=120)
    DBP: float = Field(..., gt=0)
    Temp: float = Field(..., gt=30, lt=45)
    Glucose: float = Field(..., ge=0)
    Potassium: float = Field(..., gt=0, lt=10)
    Hct: float = Field(..., ge=0, le=100)
    FiO2: float = Field(..., ge=0.21, le=1.0)
    Hgb: float = Field(..., gt=0)
    pH: float = Field(..., gt=6.5, lt=8.0)
    BUN: float = Field(..., ge=0)
    WBC: float = Field(..., gt=0)
    Magnesium: float = Field(..., gt=0)
    Creatinine: float = Field(..., ge=0)
    Platelets: float = Field(..., ge=0)
    Calcium: float = Field(..., gt=0)
    PaCO2: float = Field(..., ge=0)
    BaseExcess: float
    Chloride: float = Field(..., gt=0)
    HCO3: float = Field(..., ge=0)
    Phosphate: float = Field(..., gt=0)
    EtCO2: float = Field(..., ge=0)
    SaO2: float = Field(..., ge=0, le=100)
    PTT: float = Field(..., gt=0)
    Lactate: float = Field(..., ge=0)
    AST: float = Field(..., ge=0)
    Alkalinephos: float = Field(..., ge=0)
    Bilirubin_total: float = Field(..., ge=0)
    TroponinI: float = Field(..., ge=0)
    Fibrinogen: float = Field(..., gt=0)
    Bilirubin_direct: float = Field(..., ge=0)

    @validator('*')
    def check_nan(cls, v):
        """Ensure no NaN values in any field"""
        if isinstance(v, list):
            if any(np.isnan(x) for x in v):
                raise ValueError("List values cannot contain NaN")
        elif isinstance(v, float) and np.isnan(v):
            raise ValueError("NaN values not allowed")
        return v

    @validator('HR', 'MAP', 'O2Sat', 'SBP', 'Resp')
    def check_time_series_values(cls, v):
        """Validate time-series data"""
        if any(np.isnan(x) for x in v):
            raise ValueError("Time-series values cannot contain NaN")
        if len(v) != 10:
            raise ValueError("Exactly 10 values required for time-series features")
        return v
    
    
    


class PatientInput(BaseModel):
    patient_id: int
    probability: float
    risk_level: str
    needs_urgent_attention: bool
    shap_values: Optional[Dict[str, Dict[str, float]]] = None


class MetricsInput(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    confusion_matrix: List[List[Union[int, float]]]  # Accept int or float

class BatchInput(BaseModel):
    threshold: float
    auc: float  # top-level auc, separate from the one in metrics
    metrics: MetricsInput
    shap_values: Dict[str, Dict[str, float]]
    predictions: List[PatientInput]
    
    
    

#for the LLM generations 
class RiskAssessment(BaseModel):
    level: str = Field(default="Unknown")
    score: Union[int, float, str] = Field(default=0)
    time_frame: str = Field(default="10-hour sepsis risk")
    interpretation: str = Field(default="Assessment pending")
    detailed_analysis: str = Field(default="Detailed analysis pending")

class KeyRiskFactor(BaseModel):
    marker: str
    importance: Union[int, float] = Field(default=0.0)
    note: str = Field(default="No additional notes")

class AntibioticChoices(BaseModel):
    community_acquired: List[str] = Field(default_factory=list)
    hospital_acquired: List[str] = Field(default_factory=list)
    penicillin_allergy: List[str] = Field(default_factory=list)

class TreatmentOptions(BaseModel):
    immediate_medications: List[str] = Field(default_factory=list)
    antibiotic_choices: Union[List[str], AntibioticChoices] = Field(default_factory=lambda: AntibioticChoices())

class ClinicalGuidance(BaseModel):
    monitoring: List[str] = Field(default_factory=list)
    diagnostic_tests: List[str] = Field(default_factory=list)
    treatment_options: TreatmentOptions = Field(default_factory=TreatmentOptions)
    required_actions: List[str] = Field(default_factory=list)

class SafetyAlerts(BaseModel):
    precautions: List[str] = Field(default_factory=list)

class Disclaimers(BaseModel):
    model_use: str = Field(default="This assessment is generated by an AI model and should be used as a clinical decision support tool only.")
    physician_oversight: str = Field(default="All decisions must be validated by a qualified healthcare professional.")
    limitations: str = Field(default="This model has limitations and should not replace clinical judgment.")

class PatientReport(BaseModel):
    patient_id: int
    report_type: Optional[str] = Field(default="Sepsis Risk Assessment")
    generated_at: Optional[str] = Field(default=None)
    risk_probability: str = Field(default="0%")
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment)
    key_risk_factors: List[KeyRiskFactor] = Field(default_factory=list)
    clinical_guidance: ClinicalGuidance = Field(default_factory=ClinicalGuidance)
    safety_alerts: SafetyAlerts = Field(default_factory=SafetyAlerts)
    disclaimers: Disclaimers = Field(default_factory=Disclaimers)

class CsvPredictionResult(BaseModel):
    patient_id: int
    success: bool = True
    report: PatientReport
    risk_level: str = Field(default="Unknown")
    probability: float = Field(default=0.0)
    
    
    
# app/schemas/user.py
from pydantic import BaseModel, EmailStr, constr, model_validator
from enum import Enum


class RoleEnum(str, Enum):
    doctor = "Doctor"
    nurse = "Nurse"
    physician_assistant = "Physician Assistant"
    nurse_practitioner = "Nurse Practitioner"
    resident = "Resident"


class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    role: RoleEnum
    hospital: str
    medical_license_number: str
    password: constr(min_length=8)
    confirm_password: constr(min_length=8)

    @model_validator(mode='after')
    def check_passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self