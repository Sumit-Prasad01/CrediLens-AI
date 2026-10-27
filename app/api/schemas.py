from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    """Input features required for loan credit risk assessment."""

    person_age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant age in years (18-100)",
        examples=[30]
    )
    person_income: float = Field(
        ...,
        ge=0.0,
        description="Annual income in currency units",
        examples=[600000.0]
    )
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"] = Field(
        ...,
        description="Home ownership status",
        examples=["RENT"]
    )
    person_emp_length: float = Field(
        ...,
        ge=0.0,
        le=60.0,
        description="Employment duration in years",
        examples=[5.0]
    )
    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ] = Field(
        ...,
        description="Intent / purpose of the loan",
        examples=["PERSONAL"]
    )
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        ...,
        description="Credit risk grade assigned to loan (A = lowest risk, G = highest)",
        examples=["B"]
    )
    loan_amnt: float = Field(
        ...,
        gt=0.0,
        description="Requested loan amount",
        examples=[100000.0]
    )
    loan_int_rate: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Loan interest rate (%)",
        examples=[11.5]
    )
    loan_percent_income: float = Field(
        ...,
        ge=0.0,
        le=2.0,
        description="Loan amount as a fraction of annual income",
        examples=[0.17]
    )
    cb_person_default_on_file: Literal["Y", "N"] = Field(
        ...,
        description="Historical loan default on credit bureau file (Y/N)",
        examples=["N"]
    )
    cb_person_cred_hist_length: int = Field(
        ...,
        ge=0,
        le=60,
        description="Credit history length in years",
        examples=[6]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 30,
                "person_income": 600000.0,
                "person_home_ownership": "RENT",
                "person_emp_length": 5.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 100000.0,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.17,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6
            }
        }
    }


class PredictionResponse(BaseModel):
    """Risk assessment response structure expected by frontend and downstream services."""

    default_probability: float = Field(
        ...,
        description="Calibrated probability of default (0.0 to 1.0)",
        examples=[0.1425]
    )
    default_prediction: int = Field(
        ...,
        description="Binary decision: 0 = Low Risk / Approved, 1 = High Risk / Default Likely",
        examples=[0]
    )
    threshold: float = Field(
        ...,
        description="Operating classification decision threshold",
        examples=[0.3638]
    )
    Result: str = Field(
        ...,
        description="Verdict string displayed by the UI ('Low Risk' or 'High Risk')",
        examples=["Low Risk"]
    )
    risk_category: str = Field(
        ...,
        description="Detailed descriptive risk tier",
        examples=["Low Risk (Approved)"]
    )
    decision: str = Field(
        ...,
        description="Operational decision ('Approved' or 'Default Risk')",
        examples=["Approved"]
    )


class HealthResponse(BaseModel):
    """API health status and model availability."""

    status: str
    service: str
    model_loaded: bool
    model_path: str
    threshold: Optional[float] = None
    version: str = "1.0.0"
