from functools import lru_cache

from src.pipeline.prediction_pipeline import PredictionPipeline


@lru_cache(maxsize=1)
def get_prediction_pipeline() -> PredictionPipeline:
    return PredictionPipeline()
