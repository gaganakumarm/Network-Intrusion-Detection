from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Accepts either {"data": {...}} or a flat JSON feature object."""

    model_config = ConfigDict(extra="allow")

    data: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        if self.data is not None:
            return self.data
        return dict(self.model_extra or {})


class PredictionResponse(BaseModel):
    prediction: str
    attack_probability: float | None = None
    risk_level: str
    model_version: str
    prediction_timestamp: str


class CsvPredictionResponse(BaseModel):
    total_records: int
    attack_count: int
    normal_count: int
    prediction_table: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "Network Intrusion Detection API"


class ModelInfoResponse(BaseModel):
    model_version: str
    model_path: str
    preprocessor_path: str
    model_exists: bool
    preprocessor_exists: bool
    model_name: str | None = None
    required_feature_columns: list[str] = Field(default_factory=list)
