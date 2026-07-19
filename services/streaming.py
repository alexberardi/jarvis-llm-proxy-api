"""Streaming response helpers.

Starlette's StreamingResponse never closes its body iterator — when a client
disconnects mid-stream, the generator (and everything it holds: upstream HTTP
streams, backend locks) is only finalized whenever GC gets around to it.
ClosingStreamingResponse makes teardown deterministic: the iterator's
``aclose()``/``close()`` runs in a ``finally`` as soon as streaming stops,
so a generator's own ``finally`` block is a reliable place for cleanup.
"""

from starlette.responses import StreamingResponse
from starlette.types import Send


class ClosingStreamingResponse(StreamingResponse):
    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            # Works under pending cancellation as long as the generator's
            # finally block does not await: aclose() throws GeneratorExit at
            # the current yield and returns without hitting a checkpoint.
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()
