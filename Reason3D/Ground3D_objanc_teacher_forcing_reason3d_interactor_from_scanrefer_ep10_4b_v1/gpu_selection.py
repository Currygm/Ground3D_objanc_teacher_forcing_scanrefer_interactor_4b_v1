"""Resolve nvidia-smi physical card indices into stable CUDA UUIDs."""
import subprocess
import sys


def resolve_gpu_ids(requested, inventory):
    by_index = {}
    for line in inventory.splitlines():
        if not line.strip():continue
        index, uuid = [part.strip() for part in line.split(",",1)]
        by_index[index] = uuid
    uuids = set(by_index.values())
    selected = []
    for token in requested.split(","):
        token = token.strip()
        uuid = by_index.get(token,token if token in uuids else None)
        if uuid is None or uuid in selected:
            raise ValueError(f"Invalid or duplicate physical GPU selection: {token!r}")
        selected.append(uuid)
    return selected


if __name__ == "__main__":
    inventory = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    print(",".join(resolve_gpu_ids(sys.argv[1],inventory)))
