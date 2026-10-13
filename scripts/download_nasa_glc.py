"""Download and validate the NASA Global Landslide Catalog historical export.

The official export is governed by its terms and conditions. This script does
not bypass an access prompt or assert a licence that the catalog does not state.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://data.nasa.gov/api/views/dd9e-wu2v/rows.csv?accessType=DOWNLOAD"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "public" / "nasa_global_landslide_catalog.csv"
CATALOG_URL = "https://catalog.data.gov/dataset/global-landslide-catalog-export"


def validate_catalog_csv(csv_bytes: bytes) -> tuple[int, list[str]]:
    """Validate that a CSV looks like an event inventory, not an FoS dataset."""
    try:
        text = csv_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        fields = [name.strip().lower() for name in (reader.fieldnames or [])]
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Downloaded resource is not a readable UTF-8 CSV: {exc}") from exc

    has_coordinates = any(name in fields for name in ("latitude", "lat")) and any(
        name in fields for name in ("longitude", "lon", "lng")
    )
    has_event_date = any(name in fields for name in ("event_date", "date", "eventdate"))
    if not has_coordinates or not has_event_date:
        raise ValueError(
            "CSV does not have the expected landslide event date and latitude/longitude fields."
        )
    if not rows:
        raise ValueError("Downloaded catalog contains no data records.")
    return len(rows), fields


def download_catalog(url: str = DEFAULT_URL, output_path: Path = DEFAULT_OUTPUT) -> int:
    """Fetch the catalog export, validate it, then atomically persist the bytes."""
    request = Request(url, headers={"User-Agent": "TALUS-research-data-loader/1.0"})
    try:
        with urlopen(request, timeout=45) as response:
            content = response.read()
    except URLError as exc:
        raise RuntimeError(
            f"Could not download the NASA catalog from {url}. The NASA export may be "
            f"unavailable or may require accepting its terms at {CATALOG_URL}. "
            "No substitute or synthetic records were written. "
            f"Underlying error: {exc}"
        ) from exc

    count, _ = validate_catalog_csv(content)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temp_path.write_bytes(content)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Official export URL after access terms are satisfied.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = download_catalog(args.url, args.output)
    print(f"Saved {records} public landslide inventory records to {args.output}")
    print("This inventory has no slope FoS or mine-safety class labels; it is not used for model validation.")


if __name__ == "__main__":
    main()
