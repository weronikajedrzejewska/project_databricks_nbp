import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class NbpRate(BaseModel):
    currency: str
    code: str
    mid: float


class NbpTable(BaseModel):
    table: Literal["A", "B", "C"]
    effectiveDate: date
    rates: list[NbpRate]


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def create_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBP table into bronze JSONL.")
    parser.add_argument("--table", default="A", choices=["A", "B", "C"])
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument("--base-url", default="https://api.nbp.pl/api")
    parser.add_argument("--output-jsonl", default="data/bronze/nbp_raw.jsonl")
    args = parser.parse_args()

    if args.start_date > args.end_date:
        raise ValueError("--start-date must be <= --end-date")

    url = (
        f"{args.base_url}/exchangerates/tables/{args.table}/"
        f"{args.start_date.isoformat()}/{args.end_date.isoformat()}/?format=json"
    )

    session = create_session()
    response = session.get(url, timeout=30)
    response.raise_for_status()

    tables = [NbpTable.model_validate(item) for item in response.json()]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ingestion_ts = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as f:
        for t in tables:
            row = {
                "ingestion_ts": ingestion_ts,
                "source_url": url,
                "table_type": t.table,
                "effective_date": t.effectiveDate.isoformat(),
                "raw_payload": t.model_dump_json(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    session.close()
    print(f"OK: wrote {len(tables)} rows to {output_path}")


if __name__ == "__main__":
    main()
