"""Adapter API routes.

Endpoints for adapter-related operations like date key vocabulary
and date key adapter management.
"""

from fastapi import APIRouter

from services.date_keys import get_date_keys_response, is_adapter_trained
from services.date_key_adapter import list_available_adapters

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


@router.get("/date-keys/adapters")
async def get_date_key_adapters():
    """List all available date key adapters.

    Returns installed adapters with metadata (eval accuracy, size, etc.).
    Used by the admin panel to populate the adapter selector dropdown.
    """
    adapters = list_available_adapters()
    return {
        "adapters": adapters,
        "installed_count": sum(1 for a in adapters if a.get("installed")),
        "total_count": len(adapters),
    }
