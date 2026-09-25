#!/usr/bin/env python3
"""Download at most five selected PRADAN products per sensor using an env cookie."""

from __future__ import annotations

import os
import argparse
from pathlib import Path, PurePosixPath
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

BASE_URL = "https://pradan.issdc.gov.in"
OUTPUT_ROOT = Path("data/raw2")
CHUNK_BYTES = 8 * 1024 * 1024
MAX_RETRIES = 5
RETRY_SECONDS = 30
BETWEEN_FILES_SECONDS = 2


class AuthenticationRequired(RuntimeError):
    """Raised when PRADAN redirects a download to its login flow."""

PRODUCT_PATHS = [
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/ch2_ohr_ncp_20260103T1203563771_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/ch2_ohr_ncp_20260102T1819015920_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/ch2_ohr_ncp_20260102T2017444613_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/ch2_ohr_ncp_20260102T1224107393_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/ch2_ohr_ncp_20260102T1422564520_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260130/ch2_ohr_ncp_20260130T1309574436_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260130/ch2_ohr_ncp_20260130T1708191265_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260130/ch2_ohr_ncp_20260130T1908101751_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260130/ch2_ohr_ncp_20260130T1110042036_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260129/ch2_ohr_ncp_20260129T1117178849_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260129/ch2_ohr_ncp_20260129T1317196484_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260129/ch2_ohr_ncp_20260129T1715285977_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260129/ch2_ohr_ncp_20260129T1915291096_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20251227/ch2_ohr_ncp_20251227T1226047679_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20251227/ch2_ohr_ncp_20251227T1027178560_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20251227/ch2_ohr_ncp_20251227T1622437440_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20251227/ch2_ohr_ncp_20251227T1423592391_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/raw/20260103/ch2_ohr_nrp_20260103T1005176450_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/raw/20260103/ch2_ohr_nrp_20260103T0609041371_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/raw/20260103/ch2_ohr_nrp_20260103T1203563771_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/raw/20260103/ch2_ohr_nrp_20260103T0410224157_d_img_d18.zip?ohrc",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/raw/20260102/ch2_ohr_nrp_20260102T1819015920_d_img_d18.zip?ohrc",
]

TMC2_PRODUCT_PATHS = [
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260823/ch2_tmc_ncn_20260823T1657513854_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260823/ch2_tmc_nca_20260823T1657513854_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260823/ch2_tmc_ncf_20260823T1657513886_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260823/ch2_tmc_nrn_20260823T1657513854_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260823/ch2_tmc_nra_20260823T1657513854_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260823/ch2_tmc_nrf_20260823T1657513886_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260815/ch2_tmc_ncn_20260815T2104543018_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260815/ch2_tmc_nca_20260815T2104543018_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260815/ch2_tmc_ncf_20260815T2104543051_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260815/ch2_tmc_nrn_20260815T2104543018_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260815/ch2_tmc_nra_20260815T2104543018_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260815/ch2_tmc_nrf_20260815T2104543051_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260902/ch2_tmc_ncn_20260902T0114079604_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260902/ch2_tmc_nca_20260902T0114079637_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260902/ch2_tmc_ncf_20260902T0114079604_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260902/ch2_tmc_nrn_20260902T0114079604_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260902/ch2_tmc_nra_20260902T0114079637_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260902/ch2_tmc_nrf_20260902T0114079604_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260831/ch2_tmc_ncn_20260831T0617121701_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260831/ch2_tmc_nca_20260831T0617121701_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260831/ch2_tmc_ncf_20260831T0617121701_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260831/ch2_tmc_nrn_20260831T0617121701_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260831/ch2_tmc_nra_20260831T0617121701_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/raw/20260831/ch2_tmc_nrf_20260831T0617121701_d_img_d18.zip?tmc2",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260814/ch2_tmc_ncn_20260814T0737226524_d_img_d18.zip?tmc2",
]

IIRS_PRODUCT_PATHS = [
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/iir_collection/data/derived/20250729/ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.zip?iirs",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/iir_collection/data/derived/20240120/ch2_iir_ndi_20240120T1432235872_d_rfl_d18_srd.zip?iirs",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/iir_collection/data/derived/20240119/ch2_iir_ndi_20240119T0318228416_d_rfl_d18_srd.zip?iirs",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/iir_collection/data/derived/20240518/ch2_iir_ndi_20240518T1906068679_d_rfl_d18_srd.zip?iirs",
    "/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/iir_collection/data/derived/20240518/ch2_iir_ndi_20240518T1709309222_d_rfl_d18_srd.zip?iirs",
]


def product_path(url_path: str) -> Path:
    """Retain the product's data-level/date structure under data/raw2/sensor."""
    clean_path = url_path.split("?", 1)[0]
    collection = next((item for item in ("ohr_collection", "tmc_collection", "iir_collection")
                       if f"/{item}/" in clean_path), None)
    if collection is None:
        raise ValueError(f"Unexpected PRADAN product path: {clean_path}")
    sensor = {"ohr_collection": "OHRC", "tmc_collection": "TMC-2",
              "iir_collection": "IIRS"}[collection]
    relative = PurePosixPath(clean_path.split(f"/{collection}/", 1)[1])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe product path: {clean_path}")
    return OUTPUT_ROOT / sensor / Path(*relative.parts)


def _request(url: str, cookie: str, resume_at: int) -> tuple[object, str, int | None]:
    headers = {"Cookie": cookie, "User-Agent": "LunaMatch PRADAN downloader"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    request = Request(url, headers=headers)
    # Bound stalled reads so interrupted archive transfers reach the retry and
    # Range-resume path instead of waiting indefinitely on a quiet connection.
    response = urlopen(request, timeout=90)
    final = urlparse(response.geturl())
    if final.hostname != "pradan.issdc.gov.in" or "login" in final.path.lower() or "auth" in final.path.lower():
        response.close()
        raise AuthenticationRequired(
            "PRADAN redirected to authentication; refresh the session and retry"
        )
    status = response.status
    if status not in (200, 206):
        response.close()
        raise RuntimeError(f"Unexpected HTTP status: {status}")
    if status == 206:
        content_range = response.headers.get("Content-Range", "")
        match = re.match(r"bytes (\d+)-\d+/(\d+|\*)$", content_range)
        if not match or int(match.group(1)) != resume_at:
            response.close()
            raise RuntimeError(f"Invalid resume response Content-Range: {content_range!r}")
        mode = "ab"
        expected_size = None if match.group(2) == "*" else int(match.group(2))
    else:
        mode = "wb"
        content_length = response.headers.get("Content-Length")
        expected_size = int(content_length) if content_length and content_length.isdigit() else None
    return response, mode, expected_size


def download_one(url_path: str, cookie: str) -> Path:
    """Resume one product, confirm ZIP signature/CRC, and atomically publish."""
    url = BASE_URL + url_path
    destination = product_path(url_path)
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_zip(destination)
        print(f"Already present and valid: {destination}")
        return destination

    for attempt in range(1, MAX_RETRIES + 1):
        resume_at = partial.stat().st_size if partial.exists() else 0
        response = None
        try:
            response, mode, expected_size = _request(url, cookie, resume_at)
            if mode == "wb":
                resume_at = 0
            downloaded = resume_at
            report_after = downloaded + 64 * 1024 * 1024
            first = True
            with response, partial.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    if first and mode == "wb" and not chunk.startswith(b"PK\x03\x04"):
                        raise RuntimeError("Response is not a ZIP archive; session may have expired")
                    first = False
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= report_after:
                        print(f"  {downloaded / 1024**2:.0f} MiB transferred", flush=True)
                        report_after = downloaded + 64 * 1024 * 1024
                output.flush()
                os.fsync(output.fileno())
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Length mismatch: expected {expected_size} bytes, got {partial.stat().st_size}"
                )
            _validate_zip(partial)
            os.replace(partial, destination)
            print(f"Downloaded and verified: {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
            return destination
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, BadZipFile) as exc:
            if response is not None:
                response.close()
            if isinstance(exc, AuthenticationRequired):
                raise
            if isinstance(exc, HTTPError) and exc.code == 416 and partial.exists():
                partial.unlink()
            print(f"Attempt {attempt}/{MAX_RETRIES} failed for {destination.name}: {exc}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_SECONDS)
    raise RuntimeError("Download attempts exhausted")


def _validate_zip(path: Path) -> None:
    with ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise BadZipFile(f"CRC failure in member {bad_member}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=("OHRC", "TMC-2", "IIRS"), default="OHRC")
    parser.add_argument("--limit", type=int,
                        help="Download up to N selected products (default: 5)")
    args = parser.parse_args()
    all_paths = {"OHRC": PRODUCT_PATHS, "TMC-2": TMC2_PRODUCT_PATHS,
                 "IIRS": IIRS_PRODUCT_PATHS}[args.sensor]
    limit = args.limit if args.limit is not None else 5
    if limit < 1 or limit > min(5, len(all_paths)):
        parser.error(f"--limit must be between 1 and {min(5, len(all_paths))}")
    cookie = os.environ.get("PRADAN_COOKIE", "").strip()
    if not cookie:
        print("Set PRADAN_COOKIE to the Cookie header from your current PRADAN session.", file=sys.stderr)
        return 2
    selected_paths = all_paths[:limit]
    print(f"Downloading {len(selected_paths)} selected {args.sensor} products to {OUTPUT_ROOT.resolve()}")
    print("Sequential downloads; session cookie is never written to disk or printed.")
    completed = []
    for index, path in enumerate(selected_paths, start=1):
        print(f"[{index}/{len(selected_paths)}] {Path(path.split('?', 1)[0]).name}")
        completed.append(download_one(path, cookie))
        if index < len(selected_paths):
            time.sleep(BETWEEN_FILES_SECONDS)
    print(f"Verified {len(completed)} {args.sensor} products under {OUTPUT_ROOT / args.sensor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
