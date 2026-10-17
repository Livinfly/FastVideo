"""Fetch the pinned public FastWan release; reuse local assets only after SHA256 verification."""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL_ID = "FastVideo/FastWan2.1-T2V-14B-Diffusers"
REVISION = "ab7d6149c10bb41ba39226e2262cd2e9ee85b88e"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ranged_download(target: Path, name: str, size: int) -> None:
    """Bounded parallel HTTP ranges; resume completed chunks and verify the final SHA256 in main."""
    import requests

    partial = target.with_suffix(target.suffix + ".partial")
    ledger = target.with_suffix(target.suffix + ".ranges.json")
    chunk_size = 64 * 1024**2
    count = (size + chunk_size - 1) // chunk_size
    done = set(json.loads(ledger.read_text())) if ledger.exists() and partial.exists() else set()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)

    def fetch(index):
        begin = index * chunk_size
        end = min(size, begin + chunk_size) - 1
        for attempt in range(4):
            try:
                url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}?download=true&chunk={index}"
                with requests.get(url, headers={"Range": f"bytes={begin}-{end}"}, stream=True,
                                  timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get(
                            "content-range") != f"bytes {begin}-{end}/{size}":
                        raise ValueError("Server did not honor the requested byte range")
                    offset = begin
                    for data in response.iter_content(4 * 1024**2):
                        if offset + len(data) > end + 1:
                            raise ValueError("Oversized range response")
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(fd, view, offset)
                            offset += written
                            view = view[written:]
                    if offset != end + 1:
                        raise ValueError("Truncated range response")
                return index
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    try:
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(fetch, i) for i in range(count) if i not in done]
            for future in as_completed(futures):
                done.add(future.result())
                temporary = ledger.with_suffix(".tmp")
                temporary.write_text(json.dumps(sorted(done)))
                temporary.replace(ledger)
                if len(done) % 16 == 0 or len(done) == count:
                    print(f"Downloaded {name}: {len(done)}/{count} chunks", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    partial.replace(target)
    ledger.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    import requests
    from huggingface_hub import hf_hub_download

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true", timeout=60)
    response.raise_for_status()
    remote = response.json()
    assert remote["sha"] == REVISION
    records = []
    files = [
        f for f in remote["siblings"] if f["rfilename"].split("/")[0] in
        {"model_index.json", "README.md", "scheduler", "transformer", "text_encoder", "tokenizer", "vae"}
    ]
    # Reuse verified shared text encoder/VAE/tokenizer before fetching the large DiT shards.
    files.sort(key=lambda f: (f["rfilename"].startswith("transformer/"), f["rfilename"]))
    for item in files:
        name = item["rfilename"]
        target = output / name
        expected = item.get("lfs", {}).get("sha256")
        source = args.reuse / name if args.reuse else None
        if not target.exists() and expected and source and source.is_file() and source.stat().st_size == item["size"]:
            print("Verifying reusable asset:", name, flush=True)
            if digest(source) == expected:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
        print("Preparing:", name, flush=True)
        if not target.exists() and item["size"] > 1024**3:
            ranged_download(target, name, item["size"])
        elif not target.is_symlink():
            hf_hub_download(MODEL_ID, name, revision=REVISION, local_dir=output)
        actual = digest(target)
        if target.stat().st_size != item["size"] or (expected and actual != expected):
            raise ValueError(f"Asset integrity mismatch: {name}")
        records.append(
            dict(file=name,
                 size=item["size"],
                 sha256=actual,
                 lfs_sha256=expected,
                 reused_from=str(target.resolve()) if target.is_symlink() else None))
        record = dict(model_id=MODEL_ID, revision=REVISION, status="preparing", files=records)
        (output / "download_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    record["status"] = "complete"
    (output / "download_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Verified snapshot:", output, flush=True)


if __name__ == "__main__":
    main()
