from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.routes import router
from src.utils.exception import NetworkSecurityException
from src.utils.logger import logger


app = FastAPI(
    title="Network Intrusion Detection API",
    description="Prediction backend for the Network Intrusion Detection System.",
    version="1.0.0",
)


@app.exception_handler(NetworkSecurityException)
async def network_security_exception_handler(
    request: Request,
    exc: NetworkSecurityException,
) -> JSONResponse:
    logger.exception("Network security error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


app.include_router(router)
