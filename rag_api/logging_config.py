import logging
import contextvars
from .config import LOG_LEVEL

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True

def setup_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        #format="%(asctime)s %(levelname)s [%(name)s] [rid=%(request_id)s] %(message)s",
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    root = logging.getLogger()
    root.addFilter(RequestIdFilter())

def get_logger(name: str = "rag_api") -> logging.Logger:
    return logging.getLogger(name)

