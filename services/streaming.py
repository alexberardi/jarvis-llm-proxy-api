"""Streaming response helpers.

Starlette's StreamingResponse never closes its body iterator — when a client
disconnects mid-stream, the generator (and everything it holds: upstream HTTP
streams, backend locks) is only finalized whenever GC gets around to it.
ClosingStreamingResponse makes teardown deterministic: the iterator's
``aclose()``/``close()`` runs in a ``finally`` as soon as streaming stops,
so a generator's own ``finally`` block is a reliable place for cleanup.

One hole aclose() cannot cover: an async generator that was never started
(client disconnected before the first ``__anext__``) is closed WITHOUT its
body ever running, so cleanup living in the generator's ``finally`` is
silently skipped. ``on_teardown`` exists for that case — a synchronous,
idempotent callback invoked unconditionally after streaming stops, whether
the body ran or not. Keep it sync: it runs under a pending cancellation,
where any await would re-raise before finishing.
"""

from typing import Callable, Optional

from starlette.responses import StreamingResponse
from starlette.types import Send


class ClosingStreamingResponse(StreamingResponse):
    def __init__(
        self,
        *args,
        on_teardown: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_teardown = on_teardown

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            try:
                # Works under pending cancellation as long as the generator's
                # finally block does not await: aclose() throws GeneratorExit
                # at the current yield and returns without a checkpoint.
                aclose = getattr(self.body_iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
            finally:
                if self._on_teardown is not None:
                    self._on_teardown()
