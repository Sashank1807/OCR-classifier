"""
Volume / sustained-throughput test.

This deployment takes bulk batches rather than many concurrent users, so what
matters is: how many documents an hour does it drain, does the queue stay
orderly, does anything fail or leak, and what does a document cost on disk.

    python tools/volume_test.py --count 60 --batch 20

Submits real documents through the running HTTP API (so auth, rate limiting
and the worker pool are all exercised), polls until every job reaches a
terminal state, and samples process memory and CPU throughout.
"""

import argparse
import glob
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_key() -> str:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    raise SystemExit("No API_KEYS in .env")


def _sampler(stop_evt, samples, interval=2.0):
    """Record RSS and CPU of the server process while the batch drains."""
    try:
        import psutil
    except ImportError:
        return
    procs = [p for p in psutil.process_iter(["name", "cmdline"])
             if p.info["cmdline"] and any("run.py" in c for c in p.info["cmdline"])]
    if not procs:
        return
    for p in procs:
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    while not stop_evt.is_set():
        rss = cpu = 0.0
        for p in list(procs):
            try:
                rss += p.memory_info().rss
                cpu += p.cpu_percent(None)
            except Exception:
                procs.remove(p)
        samples.append({"t": time.time(), "rss_mb": rss / 1e6, "cpu_pct": cpu})
        stop_evt.wait(interval)


def _dir_bytes(path: str) -> int:
    return sum(f.stat().st_size for f in Path(path).glob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--batch", type=int, default=20, help="files per multipart request")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="outputs/volume_test_report.json")
    ap.add_argument("--corpus", default="uploads",
                    help="directory of documents to replay (use a representative "
                         "mix - the default uploads/ is dominated by stock "
                         "statements, the heaviest and now out-of-scope path)")
    args = ap.parse_args()

    key = _load_key()
    hdr = {"X-API-Key": key}

    pool = sorted(f for e in ("jpg", "jpeg", "png", "pdf", "webp")
                  for f in glob.glob(f"{args.corpus}/*.{e}"))
    if not pool:
        raise SystemExit("No documents in uploads/ to replay")
    files = [pool[i % len(pool)] for i in range(args.count)]
    total_in = sum(os.path.getsize(f) for f in files)

    print("=" * 78)
    print(f"VOLUME TEST  {args.count} documents, {args.batch} per request")
    print(f"input: {total_in/1e6:.1f} MB   started {datetime.now():%H:%M:%S}")
    print("=" * 78)

    disk_before = _dir_bytes("outputs") + _dir_bytes("uploads")
    samples, stop_evt = [], threading.Event()
    threading.Thread(target=_sampler, args=(stop_evt, samples), daemon=True).start()

    t_start = time.time()
    pending, submit_fail = {}, []

    # --- submit -----------------------------------------------------------
    for i in range(0, len(files), args.batch):
        chunk = files[i:i + args.batch]
        # The endpoint de-duplicates by filename inside a single request, so a
        # replayed corpus must present distinct names or the batch collapses.
        handles = [("files", (f"{i+j:04d}_{Path(f).name}", open(f, "rb")))
                   for j, f in enumerate(chunk)]
        try:
            r = requests.post(f"{args.base}/api/v1/ocr/process", headers=hdr,
                              files=handles,
                              data={"async_mode": "true", "project_name": "volume_test"},
                              timeout=300)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "30"))
                print(f"  rate limited on batch {i//args.batch+1}; waiting {wait}s")
                time.sleep(wait + 1)
                r = requests.post(f"{args.base}/api/v1/ocr/process", headers=hdr,
                                  files=[("files", (f"{i+j:04d}_{Path(f).name}", open(f, "rb")))
                                         for j, f in enumerate(chunk)],
                                  data={"async_mode": "true", "project_name": "volume_test"},
                                  timeout=300)
            if r.status_code != 200:
                submit_fail.append((i, r.status_code, r.text[:120]))
                continue
            body = r.json()
            entries = body.get("batch_requests") or (
                [{"request_id": body["request_id"], "filename": chunk[0]}] if body.get("request_id") else [])
            for e in entries:
                pending[e["request_id"]] = {"filename": e.get("filename"), "submitted": time.time()}
            print(f"  batch {i//args.batch+1}: submitted {len(entries)} "
                  f"({len(pending)} queued, +{time.time()-t_start:.0f}s)")
        finally:
            for _, (_, fh) in handles:
                try:
                    fh.close()
                except Exception:
                    pass

    t_submitted = time.time()
    print(f"\n  all submitted in {t_submitted-t_start:.1f}s; draining {len(pending)} job(s)...\n")

    # --- drain -------------------------------------------------------------
    done, statuses, last_report = {}, {}, time.time()
    while pending and time.time() - t_start < args.timeout:
        for rid in list(pending):
            try:
                r = requests.get(f"{args.base}/api/v1/ocr/status/{rid}", headers=hdr, timeout=30)
                if r.status_code != 200:
                    continue
                b = r.json()
                st = b.get("status", "")
                if st == "COMPLETED" or st == "FAILED":
                    info = pending.pop(rid)
                    doc = b.get("document") or {}
                    done[rid] = {
                        "filename": info["filename"],
                        "status": st,
                        "latency_s": round(time.time() - info["submitted"], 2),
                        "pipeline_s": doc.get("processing_time"),
                        "doc_type": doc.get("document_type"),
                        "confidence": (doc.get("structured_data") or {}).get("overall_confidence"),
                        "needs_review": (doc.get("structured_data") or {}).get("needs_manual_review"),
                    }
                    statuses[st] = statuses.get(st, 0) + 1
            except requests.RequestException:
                pass
        if time.time() - last_report > 20:
            el = time.time() - t_start
            rate = len(done) / el * 3600 if el else 0
            print(f"    +{el:>5.0f}s  done {len(done):>3}/{args.count}  "
                  f"remaining {len(pending):>3}  ~{rate:.0f} docs/hour")
            last_report = time.time()
        if pending:
            time.sleep(2)

    t_end = time.time()
    stop_evt.set()
    time.sleep(0.3)

    # --- report -------------------------------------------------------------
    wall = t_end - t_start
    lat = [d["latency_s"] for d in done.values()]
    pipe = [d["pipeline_s"] for d in done.values() if d.get("pipeline_s")]
    disk_after = _dir_bytes("outputs") + _dir_bytes("uploads")
    peak_rss = max((s["rss_mb"] for s in samples), default=0)
    mean_cpu = statistics.mean([s["cpu_pct"] for s in samples]) if samples else 0
    peak_cpu = max((s["cpu_pct"] for s in samples), default=0)

    def pct(v, p):
        return round(statistics.quantiles(v, n=100)[p - 1], 1) if len(v) > 2 else (round(max(v), 1) if v else 0)

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "documents": args.count,
        "batch_size": args.batch,
        "input_mb": round(total_in / 1e6, 1),
        "submitted_in_s": round(t_submitted - t_start, 1),
        "wall_clock_s": round(wall, 1),
        "completed": statuses.get("COMPLETED", 0),
        "failed": statuses.get("FAILED", 0),
        "unfinished": len(pending),
        "submit_errors": submit_fail,
        "throughput_docs_per_hour": round(len(done) / wall * 3600, 1) if wall else 0,
        "latency_s": {"min": round(min(lat), 1) if lat else 0,
                      "median": round(statistics.median(lat), 1) if lat else 0,
                      "p90": pct(lat, 90), "max": round(max(lat), 1) if lat else 0},
        "pipeline_s": {"median": round(statistics.median(pipe), 1) if pipe else 0,
                       "max": round(max(pipe), 1) if pipe else 0},
        "peak_rss_mb": round(peak_rss, 1),
        "cpu_pct": {"mean": round(mean_cpu, 1), "peak": round(peak_cpu, 1)},
        "disk_growth_mb": round((disk_after - disk_before) / 1e6, 1),
        "disk_per_doc_mb": round((disk_after - disk_before) / 1e6 / max(1, len(done)), 2),
        "doc_types": dict(Counter(d["doc_type"] for d in done.values())),
        "needs_review": sum(1 for d in done.values() if d.get("needs_review")),
        "results": list(done.values()),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"  completed        {report['completed']}/{args.count}   failed {report['failed']}   "
          f"unfinished {report['unfinished']}")
    print(f"  wall clock       {wall:.0f}s for {args.count} documents")
    print(f"  THROUGHPUT       {report['throughput_docs_per_hour']:.0f} documents/hour")
    print(f"  latency (queue+process)  median {report['latency_s']['median']}s  "
          f"p90 {report['latency_s']['p90']}s  max {report['latency_s']['max']}s")
    print(f"  pipeline only    median {report['pipeline_s']['median']}s  max {report['pipeline_s']['max']}s")
    print(f"  peak RSS         {report['peak_rss_mb']:.0f} MB    CPU mean {report['cpu_pct']['mean']:.0f}% "
          f"peak {report['cpu_pct']['peak']:.0f}%")
    print(f"  disk growth      {report['disk_growth_mb']} MB  ({report['disk_per_doc_mb']} MB/document)")
    print(f"  flagged for review  {report['needs_review']}/{len(done)}")
    print(f"\n  -> {args.out}")
    return 0 if not pending and not report["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
