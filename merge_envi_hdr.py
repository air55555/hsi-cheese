#!/usr/bin/env python3
"""
Merge ENVI header metadata from a "long" reference .hdr into a "target" .hdr.

Rules:
  - samples, lines, bands are always taken from the target header (never overwritten).
  - Wavelength and fwhm are copied from the long header and subset using 1-based
    inclusive band indices (defaults: BAND_START=13, BAND_END=127).
  - Other fields can be pulled from the long header via --from-long (repeatable).

Standalone: only the Python standard library.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_hdr(text: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse ENVI-style header text into ordered key -> raw_value (no leading/trailing
    key whitespace). Preserves entry order in `order`.
    Values include everything after '=' (including multi-line { ... } blocks).
    """
    lines = text.splitlines()
    i = 0
    entries: Dict[str, str] = {}
    order: List[str] = []

    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)([^=]+?)\s*=\s*(.*)$", line)
        if not m:
            i += 1
            continue

        key = m.group(2).strip()
        value_lines = [m.group(3)]
        depth = value_lines[0].count("{") - value_lines[0].count("}")
        j = i + 1
        while j < len(lines) and depth > 0:
            value_lines.append(lines[j])
            depth += lines[j].count("{") - lines[j].count("}")
            j += 1

        raw_value = "\n".join(value_lines).rstrip()
        entries[key] = raw_value
        order.append(key)
        i = j

    return entries, order


def _strip_outer_braces(value: str) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    return value


def parse_numeric_list(value: str) -> List[float]:
    inner = _strip_outer_braces(value)
    parts = re.split(r"[,\s]+", inner)
    out: List[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def format_envi_list(values: List[float], indent: str = "  ") -> str:
    lines = ["{"]
    for v in values:
        if v == int(v):
            lines.append(f"{indent}{int(v)},")
        else:
            lines.append(f"{indent}{v},")
    lines.append("}")
    return "\n".join(lines)


def subset_bands(values: List[float], start_1: int, end_1: int) -> List[float]:
    if start_1 < 1 or end_1 < start_1:
        raise ValueError(f"Invalid band range {start_1}..{end_1} (1-based inclusive)")
    i0 = start_1 - 1
    i1 = end_1  # exclusive slice end
    if i1 > len(values):
        raise ValueError(
            f"Band end {end_1} exceeds list length {len(values)} "
            f"(subset needs indices within 1..{len(values)})"
        )
    return values[i0:i1]


def merge_headers(
    target: Dict[str, str],
    target_order: List[str],
    long: Dict[str, str],
    band_start: int,
    band_end: int,
    extra_long_keys: List[str],
) -> Tuple[Dict[str, str], List[str]]:
    merged = dict(target)
    order = list(target_order)

    wl_key = "Wavelength"
    fwhm_key = "fwhm"

    if wl_key in long:
        wl = subset_bands(parse_numeric_list(long[wl_key]), band_start, band_end)
        merged[wl_key] = format_envi_list(wl)
        if wl_key not in order:
            order.append(wl_key)

    if fwhm_key in long:
        fw = subset_bands(parse_numeric_list(long[fwhm_key]), band_start, band_end)
        merged[fwhm_key] = format_envi_list(fw)
        if fwhm_key not in order:
            order.append(fwhm_key)

    protected = {"samples", "lines", "bands"}
    for k in extra_long_keys:
        if k in protected:
            continue
        if k in long:
            merged[k] = long[k]
            if k not in order:
                order.append(k)

    return merged, order


def write_hdr(path: Path, entries: Dict[str, str], order: List[str]) -> None:
    lines_out: List[str] = []
    seen = set()
    for k in order:
        if k not in entries or k in seen:
            continue
        seen.add(k)
        v = entries[k]
        if "\n" in v or v.lstrip().startswith("{"):
            lines_out.append(f"{k} = {v}")
        else:
            lines_out.append(f"{k} = {v}")
        lines_out.append("")
    for k, v in entries.items():
        if k in seen:
            continue
        lines_out.append(f"{k} = {v}")
        lines_out.append("")

    text = "\n".join(lines_out).rstrip() + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge ENVI .hdr metadata (text) long -> target.")
    ap.add_argument("--target", required=True, type=Path, help="HDR to keep layout/bands dims from")
    ap.add_argument("--long", required=True, type=Path, help="HDR to take Wavelength/fwhm (+ extras) from")
    ap.add_argument("--output", required=True, type=Path, help="Written merged HDR path")
    ap.add_argument("--band-start", type=int, default=13, help="1-based first band index (in long cube)")
    ap.add_argument("--band-end", type=int, default=127, help="1-based last band index (inclusive)")
    ap.add_argument(
        "--from-long",
        action="append",
        default=[],
        help=(
            "Additional key name to copy from long (repeatable). "
            "samples, lines, bands are never copied from long."
        ),
    )
    args = ap.parse_args()

    tgt_text = args.target.read_text(encoding="utf-8", errors="replace")
    long_text = args.long.read_text(encoding="utf-8", errors="replace")

    target_entries, target_order = parse_hdr(tgt_text)
    long_entries, _ = parse_hdr(long_text)

    # Prefer target's description / default bands unless you pass e.g.
    # --from-long description  or  --from-long default bands
    default_extras = [
        "sensor type",
        "acquisition date",
        "Start Time",
        "Stop Time",
        "binning",
        "fps",
        "fps_qpf",
        "tint",
        "trigger mode",
        "sensorid",
        "VNIR temperature",
        "temperature",
    ]
    extra = list(dict.fromkeys(default_extras + args.from_long))

    merged, order = merge_headers(
        target_entries,
        target_order,
        long_entries,
        args.band_start,
        args.band_end,
        extra,
    )

    n_bands = int(merged["bands"]) if merged.get("bands") else None
    if n_bands is not None and "Wavelength" in merged:
        got = len(parse_numeric_list(merged["Wavelength"]))
        if got != n_bands:
            print(
                f"Warning: target bands={n_bands} but subset Wavelength length={got} "
                f"(check --band-start/--band-end vs target).",
                file=sys.stderr,
            )
    if n_bands is not None and "fwhm" in merged:
        got = len(parse_numeric_list(merged["fwhm"]))
        if got != n_bands:
            print(
                f"Warning: target bands={n_bands} but subset fwhm length={got}.",
                file=sys.stderr,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_hdr(args.output, merged, order)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
