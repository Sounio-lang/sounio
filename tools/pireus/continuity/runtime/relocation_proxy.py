#!/usr/bin/env python3
"""Transparent PostgreSQL TCP forwarding; credentials and payloads are never logged."""
import argparse
import asyncio
import contextlib
import signal

async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen-host", action="append")
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--target-host", required=True)
    ap.add_argument("--target-port", type=int, default=5432)
    ap.add_argument("--source-host")
    args = ap.parse_args()
    active = set()
    async def pump(reader, writer):
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
            await writer.drain()
    async def connection(reader, writer):
        task = asyncio.current_task()
        active.add(task)
        upstream = None
        pumps = []
        try:
            other_reader, upstream = await asyncio.wait_for(
                asyncio.open_connection(args.target_host, args.target_port,
                                        local_addr=(args.source_host, 0) if args.source_host else None), timeout=5)
            pumps = [asyncio.create_task(pump(reader, upstream)),
                     asyncio.create_task(pump(other_reader, writer))]
            await asyncio.gather(*pumps)
        except (OSError, asyncio.TimeoutError):
            # Connection failure closes the client socket; no query or credential logging.
            pass
        finally:
            for pending in pumps:
                if not pending.done():
                    pending.cancel()
            if pumps:
                await asyncio.gather(*pumps, return_exceptions=True)
            for stream in [writer, upstream]:
                if stream is not None:
                    stream.close()
                    with contextlib.suppress(OSError):
                        await stream.wait_closed()
            active.discard(task)
    server = await asyncio.start_server(connection, args.listen_host or ["0.0.0.0", "::"], args.listen_port)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    print("POSTGRES_FORWARDER_READY", flush=True)
    await stop.wait()
    server.close()
    await server.wait_closed()
    if active:
        _, pending = await asyncio.wait(active, timeout=10)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
