from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import UTC, datetime

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.dependencies import get_prediction_pipeline
from api.schemas import (
    CsvPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import load_object
from src.utils.exception import NetworkSecurityException
from src.utils.logger import logger


router = APIRouter()
MODEL_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
) -> ModelInfoResponse:
    model_path = Path(pipeline.model_path)
    preprocessor_path = Path(pipeline.preprocessor_path)
    model_name = None

    if model_path.exists():
        try:
            model_bundle = load_object(model_path)
            if isinstance(model_bundle, dict):
                model_name = model_bundle.get("model_name")
            else:
                model_name = model_bundle.__class__.__name__
        except NetworkSecurityException:
            logger.exception("Unable to load model info from %s", model_path)

    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        model_path=str(model_path),
        preprocessor_path=str(preprocessor_path),
        model_exists=model_path.exists(),
        preprocessor_exists=preprocessor_path.exists(),
        model_name=model_name,
        required_feature_columns=pipeline.required_feature_columns,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
) -> PredictionResponse:
    record = request.to_record()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prediction input cannot be empty",
        )

    dataframe = pd.DataFrame([record])
    _validate_required_columns(dataframe, pipeline.required_feature_columns)

    prediction_table = pipeline.predict(dataframe)
    row = prediction_table.iloc[0].to_dict()

    return PredictionResponse(
        prediction=str(row["prediction"]),
        attack_probability=_optional_float(row.get("attack_probability")),
        risk_level=str(row["risk_level"]),
        model_version=MODEL_VERSION,
        prediction_timestamp=_utc_timestamp(),
    )


@router.post("/predict-csv", response_model=CsvPredictionResponse)
async def predict_csv(
    file: UploadFile = File(...),
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
) -> CsvPredictionResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a CSV file",
        )

    uploaded_bytes = await file.read()
    if not uploaded_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV is empty",
        )

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(uploaded_bytes)
            temp_path = Path(temp_file.name)

        input_dataframe = pd.read_csv(temp_path)
        _validate_required_columns(input_dataframe, pipeline.required_feature_columns)

        prediction_table = pipeline.predict_from_csv(temp_path)
        records = prediction_table.to_dict(orient="records")
        predictions = prediction_table["prediction"].astype(str).str.lower()

        return CsvPredictionResponse(
            total_records=len(prediction_table),
            attack_count=int((predictions == "attack").sum()),
            normal_count=int((predictions == "normal").sum()),
            prediction_table=records,
        )
    except pd.errors.EmptyDataError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV has no readable data",
        ) from error
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _validate_required_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required feature columns: {missing_columns}",
        )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
