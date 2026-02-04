"""Adapter API routes.

Endpoints for adapter-related operations like date key vocabulary.
"""

from fastapi import APIRouter

from services.date_keys import get_date_keys_response, is_adapter_trained

router = APIRouter(prefix="/v1/adapters", tags=["adapters"])


@router.get("/date-keys")
async def get_date_keys():
    """Get the vocabulary of supported date keys.

    This endpoint returns all semantic date keys that the date extraction
    adapter can recognize and output. Consumers use this to:
    - Validate their date context dictionary coverage
    - Generate client-side constants
    - Stay in sync with supported keys

    This endpoint is unauthenticated - it only exposes vocabulary, no sensitive data.
    """
    response = get_date_keys_response()
    response["adapter_trained"] = is_adapter_trained()
    return response
