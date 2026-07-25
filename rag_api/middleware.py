import time
import uuid
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from .logging_config import request_id_ctx, get_logger

logger = get_logger(__name__)

class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get(self.header_name) or str(uuid.uuid4())
        token = request_id_ctx.set(rid)

        start = time.perf_counter()
        client_ip: Optional[str] = request.client.host if request.client else None
        response: Optional[Response] = None

        try:
            response = await call_next(request)
            return response
        except Exception:
            logger.exception(
                "Unhandled exception: method=%s path=%s client_ip=%s",
                request.method, request.url.path, client_ip
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            status = response.status_code if response else "ERR"

            logger.info(
                "HTTP %s %s status=%s duration_ms=%.2f client_ip=%s",
                request.method, request.url.path, status, duration_ms, client_ip
            )

            if response:
                response.headers[self.header_name] = rid

            request_id_ctx.reset(token)
