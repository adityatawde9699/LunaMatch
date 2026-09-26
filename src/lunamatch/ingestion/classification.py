"""Conservative sensor classification from product provenance."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def classify_sensor(source: str | Path) -> dict[str, Any]:
    """Classify a product from its path, returning evidence instead of guessing."""
    path = Path(source)
    text = str(path).lower()
    if "synthetic" in text or "sample" in path.parts:
        return {"sensor": "UNKNOWN", "confidence": 0.0,
                "evidence": "synthetic/development sample; no instrument provenance",
                "source": str(path)}
    matches = []
    if re.search(r"(^|[^a-z])ohrc?([^a-z]|$)|ch2_ohr", text):
        matches.append(("OHRC", "OHRC/OHR product identifier"))
    if re.search(r"(^|[^a-z])tmc(?:-?2)?([^a-z]|$)|ch2_tmc", text):
        matches.append(("TMC-2", "TMC product identifier"))
    if re.search(r"(^|[^a-z])iirs?([^a-z]|$)|ch2_iir", text):
        matches.append(("IIRS", "IIRS/IIR product identifier"))
    sensors = {item[0] for item in matches}
    if len(sensors) == 1:
        sensor, evidence = matches[0]
        return {"sensor": sensor, "confidence": 1.0, "evidence": evidence,
                "source": str(path)}
    if len(sensors) > 1:
        evidence = "conflicting instrument identifiers"
    else:
        evidence = "no authoritative instrument identifier in path"
    return {"sensor": "UNKNOWN", "confidence": 0.0, "evidence": evidence,
            "source": str(path)}
