"""Atomic shared-filesystem evaluation aggregation; no distributed collectives."""
from pathlib import Path
import json
import os
import time
import uuid


def publish_shard(directory, run_id, rank, world_size, payload):
    if not 0 <= rank < world_size:
        raise ValueError("Invalid evaluation rank")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"rank_{rank:05d}.json"
    temporary = directory / f".rank_{rank:05d}.{uuid.uuid4().hex}.tmp"
    envelope = {"format": 1, "run_id": run_id, "rank": rank,
                "world_size": world_size, "payload": payload}
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(envelope, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def wait_for_shards(directory, run_id, world_size, *, timeout=86400,
                    poll_interval=5, report_interval=60, report=print):
    if timeout <= 0 or poll_interval <= 0 or report_interval <= 0 or world_size <= 0:
        raise ValueError("Shard wait limits and world size must be positive")
    directory = Path(directory)
    start = time.monotonic()
    deadline = start + timeout
    next_report = start
    payloads = {}
    while len(payloads) < world_size:
        for rank in range(world_size):
            if rank in payloads:
                continue
            path = directory / f"rank_{rank:05d}.json"
            try:
                with path.open(encoding="utf-8") as handle:
                    value = json.load(handle)
            except FileNotFoundError:
                continue
            expected = {"format": 1, "run_id": run_id, "rank": rank, "world_size": world_size}
            if any(value.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Evaluation shard metadata mismatch: {path}")
            payloads[rank] = value["payload"]
        if len(payloads) == world_size:
            break
        now = time.monotonic()
        missing = ", ".join(str(r) for r in range(world_size) if r not in payloads)
        if now >= deadline:
            raise TimeoutError(f"Evaluation results timed out after {timeout}s; missing ranks: {missing}; shards: {directory}")
        if now >= next_report:
            report(f"Waiting for evaluation ranks: {missing}; completed {len(payloads)}/{world_size}; elapsed {now-start:.0f}s")
            next_report = now + report_interval
        time.sleep(min(poll_interval, max(0, deadline-now)))
    return [payloads[r] for r in range(world_size)]
