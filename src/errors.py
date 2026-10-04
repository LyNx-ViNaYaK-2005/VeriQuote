"""Safe UI errors. Never surface provider response bodies or credentials."""
import logging

logger = logging.getLogger(__name__)


class AppError(Exception):
    pass


def provider_error(provider: str, exc: Exception) -> AppError:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    # Exception messages can contain document text, URLs, or credentials.
    logger.warning("%s failure: type=%s status=%s", provider, type(exc).__name__, status)
    if status in (401, 403):
        detail = "Check the server API key and its permissions."
    elif status == 429:
        detail = "The request limit or credit allowance was reached. Wait and try again, or check your provider account."
    elif status in (400, 404, 422):
        detail = "Check that the configured model supports this API and is available from the selected provider."
    elif "timeout" in type(exc).__name__.lower():
        detail = "The request timed out. Please try again."
    else:
        detail = "The service could not complete the request. Check the network and provider status, then try again."
    return AppError(f"{provider}: {detail}")
