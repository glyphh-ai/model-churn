#!/usr/bin/env python3
"""
Seed the churn model with demo customer records.

Loads 25 synthetic customer accounts from demo/customers.jsonl into the
running model via the listener API. Run this after `glyphh dev . -d`.

Usage:
    python seed_demo.py                          # localhost:8002 (default)
    python seed_demo.py --url http://host:8002   # custom server
    python seed_demo.py --org my-org             # custom org (default: local-dev-org)
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

DEMO_FILE = Path(__file__).parent / "demo" / "customers.jsonl"
DEFAULT_URL = "http://localhost:8002"
DEFAULT_ORG = "local-dev-org"
MODEL_ID = "churn"


def load_demo_records() -> list[dict]:
    records = []
    with open(DEMO_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def post_records(base_url: str, org_id: str, records: list[dict]) -> dict:
    url = f"{base_url}/{org_id}/{MODEL_ID}/listener"
    payload = json.dumps({"records": records, "batch_size": 25}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wait_for_job(base_url: str, job_id: str, timeout: int = 30) -> None:
    import time
    url = f"{base_url}/jobs/{job_id}/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                status = data.get("status", "")
                if status in ("completed", "failed", "done"):
                    if status == "failed":
                        print(f"  Warning: job reported failure: {data}")
                    return
        except Exception:
            pass
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed churn model with demo customers")
    parser.add_argument("--url", default=DEFAULT_URL, help="Glyphh server URL")
    parser.add_argument("--org", default=DEFAULT_ORG, help="Organization ID")
    args = parser.parse_args()

    print(f"Loading demo records from {DEMO_FILE.relative_to(Path.cwd())}...")
    records = load_demo_records()
    print(f"  {len(records)} customers found")

    print(f"\nPosting to {args.url}/{args.org}/{MODEL_ID}/listener ...")
    try:
        resp = post_records(args.url, args.org, records)
    except urllib.error.URLError as e:
        print(f"\nError: could not reach server — is `glyphh dev . -d` running?")
        print(f"  {e}")
        sys.exit(1)

    job_id = resp.get("job_id")
    total = resp.get("total_records", len(records))
    print(f"  Queued {total} records (job {job_id})")

    if job_id:
        print("  Waiting for ingestion to complete...")
        wait_for_job(args.url, str(job_id))

    print(f"\nDone. {total} demo customers loaded into the churn model.")
    print("\nTry these queries in `glyphh chat`:")
    print("  what customers are likely to churn in the next 30 days?")
    print("  show me accounts with no logins")
    print("  who has excessive support tickets?")
    print("  show me healthy growing accounts")
    print("  which customers should I prioritize this week?")


if __name__ == "__main__":
    main()
