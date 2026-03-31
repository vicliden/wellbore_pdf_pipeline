"""
Clean wellbore PDF pipeline.

Flow:
1) Load CSV rows and keep all columns.
2) Group documents by priority (high, fallback, skip).
3) Detect probable text layer via partial PDF download.
4) If text-layer PDF: download full PDF, extract text with pypdf, send text to Claude Haiku.
5) If scanned/non-text PDF: send PDF URL to Claude Haiku first; if that fails,
   download and chunk PDF pages to stay within model limits.
6) Merge extracted rows, detect conflicts, and write output CSV.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic
import httpx
import pypdf
from dotenv import load_dotenv

load_dotenv()


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

error_handler = logging.FileHandler("errors.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(error_handler)


# Constants
MAX_PAGES_PER_CHUNK = 20
SECONDS_BETWEEN_CALLS = 5
RATE_LIMIT_WAIT_SECONDS = 60
DIAMETER_TOLERANCE_IN = 0.5
PDF_HEAD_BYTES = 65536


# Schema
@dataclass
class CasingString:
    wellbore: str
    casing_type: Optional[str] = None
    casing_diameter_in: Optional[float] = None
    casing_depth_m: Optional[float] = None
    hole_diameter_in: Optional[float] = None
    hole_depth_m: Optional[float] = None
    lot_fit_mud_equiv: Optional[float] = None
    formation_test_type: Optional[str] = None
    source_section: Optional[str] = None
    confidence: Optional[str] = None
    source_documents: list[str] = field(default_factory=list)
    conflicts: Optional[dict[str, Any]] = None


HIGH_PRIORITY = [
    "WELL_COMPLETION_REPORT",
    "WDSS_GENERAL_INFORMATION",
    "WDSS_COMPLETION_LOG",
    "INDIVIDUAL_WELL_RECORD",
    "DRILLING_MUD_RECORD",
    "DRILLING_MUD_REPORT",
    "DRILLING_FLUID_SUMMARY",
    "DRILL_STEM_TEST_DATA",
    "FORMATION_TEST",
    "EXPLORATORY_TEST",
]

FALLBACK = [
    "WELL_PRODUCTION_TEST",
    "TEMPORARY_SUSPENSION_PROCEDURE",
    "CHANGE_IN_DRILLING_PROGRAM",
]


SYSTEM_PROMPT = """You are an expert in Norwegian Continental Shelf (NCS) well documentation.
Your task is to extract casing string data from oil well PDFs from the SODIR database. Be sure to follow unit-specifications.

Casing types: conductor, surface, intermediate, production, liner
Typical casing diameters: 30\", 20\", 13 3/8\", 9 5/8\", 7\"
Expected depth ranges: 0-4000m
LOT/FIT values: typically 1.2-2.0 g/cm3
LOT/FIT unit conversion: LOT/FIT is measured in psi at a specific depth.
  To convert to mud equivalent weight (g/cm3): EMW = psi / (0.052 * depth_in_feet).
  If depth is in meters, convert first: depth_ft = depth_m * 3.281.
  If the document already expresses LOT/FIT as mud weight equivalent (g/cm3 or SG),
  use the value directly without conversion.

IMPORTANT: Each distinct casing string must be its own object in the list.
  A well typically has 3-5 casing strings with different diameters and depths.
  Do NOT merge separate casing strings into one entry.

Return ONLY raw JSON with no markdown, no code fences, no preamble. Use this exact schema:
{
  "casing_strings": [
    {
      "casing_type": string or null,
      "casing_diameter_in": float or null,
      "casing_depth_m": float or null,
      "hole_diameter_in": float or null,
      "hole_depth_m": float or null,
      "lot_fit_mud_equiv": float or null,
      "formation_test_type": string or null,
      "source_section": string or null,
      "confidence": "high" | "medium" | "low"
    }
  ],
  "relevant": boolean,
  "document_type": string or null
}

Confidence levels:
- high: value explicitly stated with clear units
- medium: value inferred from context or partially stated
- low: value ambiguous or partially legible

Handle both Norwegian and English text. Return nulls for missing fields.
Do not hallucinate values. If no relevant data is found, return an empty
casing_strings list and relevant=false. Be accurate about units and convert if needed.

Use this register of synonyms to identify fields:
{
    "Wellbore": ["Wellbore", "Well", "Borehole", "Hole", "Drillhole", "Well ID", "Well name", "UWI", "Brønn", "Borehull"],
    "Casing type": ["Casing type", "Casing string", "String type", "Tubular type", "Casing category", "Liner type", "Foringsrør", "Ankerrør", "Overflaterør", "Stigerør"],
    "Casing diameter [in]": ["Casing diameter", "Casing OD", "Outer diameter", "Nominal diameter", "Pipe size", "Casing size", "String OD", "Rørdiameter"],
    "Casing depth [m]": ["Casing depth", "Setting depth", "Casing shoe depth", "Shoe depth", "Casing TD", "String depth", "Skodybde", "Settedybde"],
    "Hole diameter [in]": ["Hole diameter", "Bit size", "Drill bit diameter", "Bore diameter", "Open hole diameter", "Section diameter", "Borestørrelse", "Bithulldiameter"],
    "Hole depth [m]": ["Hole depth", "Total depth", "TD", "Measured depth", "MD", "Drilled depth", "Borehole depth", "Section TD", "Boredybde", "Total dybde"],
    "LOT/FIT mud eqv. [g/cm3]": ["Leak-off test", "LOT", "Formation integrity test", "FIT", "Fracture gradient", "Max mud weight", "EMW", "Equivalent mud weight", "Lekkasjetest", "Formasjonsintegritetstest"],
    "Formation test type": ["Formation test type", "Test type", "Pressure test type", "Integrity test", "Shoe test", "Casing shoe test type", "Limit test type", "Testtype", "Trykktest"]
}
"""


class PipelineClean:
    def __init__(self, csv_path: str, wellbores: Optional[list[str]] = None):
        self.csv_path = Path(csv_path)
        self.wellbores = wellbores or ["7/11-1", "7/11-2", "7/11-3", "7/11-7"]
        self.client = anthropic.Anthropic()

        self.results: dict[str, list[CasingString]] = {wb: [] for wb in self.wellbores}
        self.stats = {
            "total_api_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "documents_processed": 0,
            "documents_returned_data": 0,
            "documents_failed": 0,
            "conflicts": [],
            "text_layer_true": 0,
            "text_layer_false": 0,
        }

    # CSV loading and priority grouping
    def load_csv(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        grouped = {wb: {"high": [], "fallback": [], "skip": []} for wb in self.wellbores}

        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {
                    (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                    for k, v in row.items()
                }

                wellbore = clean_row.get("wlbName") or clean_row.get("wellbore") or clean_row.get("Wellbore")
                doc_name = clean_row.get("wlbDocumentName") or clean_row.get("documentName") or clean_row.get("name", "")
                pdf_url = clean_row.get("wlbDocumentUrl") or clean_row.get("url")

                if not wellbore or not pdf_url:
                    continue
                if wellbore not in grouped:
                    continue

                priority = self._classify_priority(str(doc_name))
                grouped[wellbore][priority].append(
                    {
                        "name": str(doc_name),
                        "url": str(pdf_url),
                        "priority": priority,
                        "row": clean_row,
                    }
                )

        for wb, groups in grouped.items():
            logger.info(
                "%s: %s high priority, %s fallback, %s skipped",
                wb,
                len(groups["high"]),
                len(groups["fallback"]),
                len(groups["skip"]),
            )

        return grouped

    def _classify_priority(self, doc_name: str) -> str:
        upper = doc_name.upper()
        for pattern in HIGH_PRIORITY:
            if pattern in upper:
                return "high"
        for pattern in FALLBACK:
            if pattern in upper:
                return "fallback"
        return "skip"

    # PDF helpers
    def _fetch_pdf_head(self, url: str, byte_limit: int = PDF_HEAD_BYTES) -> Optional[bytes]:
        headers = {"Range": f"bytes=0-{byte_limit - 1}"}
        try:
            response = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
            if response.status_code in (200, 206):
                return response.content
            logger.warning("Range request failed for %s (status=%s)", url, response.status_code)
            return None
        except Exception as exc:
            logger.error("Partial PDF fetch failed for %s: %s", url, exc)
            return None

    def _has_text_layer_partial(self, pdf_head_bytes: bytes) -> Optional[bool]:
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_head_bytes))
            if not reader.pages:
                return None
            text = reader.pages[0].extract_text()
            if text and text.strip():
                return True
            return False
        except Exception:
            return None

    def quick_text_layer_check(self, url: str) -> bool:
        head_bytes = self._fetch_pdf_head(url)
        if not head_bytes:
            self.stats["text_layer_false"] += 1
            return False

        result = self._has_text_layer_partial(head_bytes)
        if result is None:
            logger.info("Partial parse inconclusive for %s, assuming no text layer", url)
            self.stats["text_layer_false"] += 1
            return False

        if result:
            self.stats["text_layer_true"] += 1
        else:
            self.stats["text_layer_false"] += 1

        return result

    def _fetch_pdf_bytes(self, url: str) -> Optional[bytes]:
        try:
            response = httpx.get(url, timeout=120, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            logger.error("Full PDF download failed for %s: %s", url, exc)
            return None

    def _extract_text_from_pdf(self, pdf_bytes: bytes, max_pages: Optional[int] = None) -> Optional[str]:
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages = reader.pages if max_pages is None else reader.pages[:max_pages]
            text_parts: list[str] = []
            for page in pages:
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text)
            return "\n".join(text_parts) if text_parts else None
        except Exception as exc:
            logger.error("Text extraction failed: %s", exc)
            return None

    def _split_pdf_chunks(self, pdf_bytes: bytes) -> list[tuple[int, int, str]]:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        if total_pages <= MAX_PAGES_PER_CHUNK:
            b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
            return [(1, total_pages, b64)]

        logger.info("PDF has %s pages, splitting into chunks of %s", total_pages, MAX_PAGES_PER_CHUNK)
        chunks: list[tuple[int, int, str]] = []
        for start in range(0, total_pages, MAX_PAGES_PER_CHUNK):
            end = min(start + MAX_PAGES_PER_CHUNK, total_pages)
            writer = pypdf.PdfWriter()
            for page_num in range(start, end):
                writer.add_page(reader.pages[page_num])
            buffer = io.BytesIO()
            writer.write(buffer)
            b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
            chunks.append((start + 1, end, b64))
        return chunks

    # Claude helpers
    def _parse_json_response(self, raw_text: str) -> Optional[dict[str, Any]]:
        raw = raw_text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error: %s", exc)
            return None

    def _extract_message_text(self, message: Any) -> str:
        text_parts: list[str] = []
        for block in getattr(message, "content", []):
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                text_parts.append(block_text)
        return "\n".join(text_parts).strip()

    def _call_claude_with_text(self, text: str, doc_name: str, model: str = "claude-haiku-4-5") -> Optional[dict[str, Any]]:
        label = f"{doc_name} [text]"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                time.sleep(SECONDS_BETWEEN_CALLS)
                message = self.client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Document name: {label}\n\n"
                                f"Extracted text from PDF:\n\n{text}\n\n"
                                "Extract all casing string data from this text. "
                                "Return only raw JSON, no markdown."
                            ),
                        }
                    ],
                )

                self.stats["total_api_calls"] += 1
                self.stats["total_input_tokens"] += message.usage.input_tokens
                self.stats["total_output_tokens"] += message.usage.output_tokens

                return self._parse_json_response(self._extract_message_text(message))

            except anthropic.RateLimitError as exc:
                logger.warning(
                    "Rate limited on %s (attempt %s/%s). Waiting %ss: %s",
                    label,
                    attempt + 1,
                    max_retries,
                    RATE_LIMIT_WAIT_SECONDS,
                    exc,
                )
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            except anthropic.BadRequestError as exc:
                logger.error("Bad request for %s: %s", label, exc)
                return None
            except Exception as exc:
                logger.error("API error for %s (attempt %s/%s): %s", label, attempt + 1, max_retries, exc)
                if attempt < max_retries - 1:
                    time.sleep(10)

        logger.error("All retries exhausted for %s", label)
        return None

    def _call_claude_with_pdf_url(self, pdf_url: str, doc_name: str, model: str = "claude-haiku-4-5") -> Optional[dict[str, Any]]:
        label = f"{doc_name} [pdf-url]"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                time.sleep(SECONDS_BETWEEN_CALLS)
                message = self.client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "url",
                                        "url": pdf_url,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        f"Document name: {label}\n"
                                        "Extract all casing string data from this document. "
                                        "Return only raw JSON, no markdown."
                                    ),
                                },
                            ],
                        }
                    ],
                )

                self.stats["total_api_calls"] += 1
                self.stats["total_input_tokens"] += message.usage.input_tokens
                self.stats["total_output_tokens"] += message.usage.output_tokens

                return self._parse_json_response(self._extract_message_text(message))

            except anthropic.RateLimitError as exc:
                logger.warning(
                    "Rate limited on %s (attempt %s/%s). Waiting %ss: %s",
                    label,
                    attempt + 1,
                    max_retries,
                    RATE_LIMIT_WAIT_SECONDS,
                    exc,
                )
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            except anthropic.BadRequestError as exc:
                logger.warning("Bad request for %s (may be size/pages): %s", label, exc)
                return None
            except Exception as exc:
                logger.error("API error for %s (attempt %s/%s): %s", label, attempt + 1, max_retries, exc)
                if attempt < max_retries - 1:
                    time.sleep(10)

        logger.error("All retries exhausted for %s", label)
        return None

    def _call_claude_with_pdf_base64(
        self,
        pdf_b64: str,
        doc_name: str,
        chunk_label: str,
        model: str = "claude-haiku-4-5",
    ) -> Optional[dict[str, Any]]:
        label = f"{doc_name} {chunk_label}".strip()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                time.sleep(SECONDS_BETWEEN_CALLS)
                message = self.client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": pdf_b64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        f"Document name: {label}\n"
                                        "Extract all casing string data from this document. "
                                        "Return only raw JSON, no markdown."
                                    ),
                                },
                            ],
                        }
                    ],
                )

                self.stats["total_api_calls"] += 1
                self.stats["total_input_tokens"] += message.usage.input_tokens
                self.stats["total_output_tokens"] += message.usage.output_tokens

                return self._parse_json_response(self._extract_message_text(message))

            except anthropic.RateLimitError as exc:
                logger.warning(
                    "Rate limited on %s (attempt %s/%s). Waiting %ss: %s",
                    label,
                    attempt + 1,
                    max_retries,
                    RATE_LIMIT_WAIT_SECONDS,
                    exc,
                )
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            except anthropic.BadRequestError as exc:
                logger.error("Bad request for %s: %s", label, exc)
                return None
            except Exception as exc:
                logger.error("API error for %s (attempt %s/%s): %s", label, attempt + 1, max_retries, exc)
                if attempt < max_retries - 1:
                    time.sleep(10)

        logger.error("All retries exhausted for %s", label)
        return None

    # Extraction routing
    def _extract_with_text_path(self, pdf_url: str, doc_name: str) -> dict[str, Any]:
        pdf_bytes = self._fetch_pdf_bytes(pdf_url)
        if not pdf_bytes:
            self.stats["documents_failed"] += 1
            return {"casing_strings": [], "relevant": False}

        text = self._extract_text_from_pdf(pdf_bytes)
        if not text:
            logger.warning("No extractable text from %s, switching to vision path", doc_name)
            return self._extract_with_vision_path(pdf_url, doc_name)

        data = self._call_claude_with_text(text, doc_name)
        if data is None:
            self.stats["documents_failed"] += 1
            return {"casing_strings": [], "relevant": False}

        return data

    def _extract_with_vision_path(self, pdf_url: str, doc_name: str) -> dict[str, Any]:
        # Preferred path for scanned PDFs: send URL directly.
        data = self._call_claude_with_pdf_url(pdf_url, doc_name)
        if data is not None:
            return data

        # Fallback when URL-based call fails (often large or complex PDFs):
        # download and chunk by pages.
        logger.info("URL-based extraction failed for %s, trying chunked upload", doc_name)

        pdf_bytes = self._fetch_pdf_bytes(pdf_url)
        if not pdf_bytes:
            self.stats["documents_failed"] += 1
            return {"casing_strings": [], "relevant": False}

        try:
            chunks = self._split_pdf_chunks(pdf_bytes)
        except Exception as exc:
            logger.error("Could not split PDF %s: %s", doc_name, exc)
            self.stats["documents_failed"] += 1
            return {"casing_strings": [], "relevant": False}

        all_casing_strings: list[dict[str, Any]] = []
        relevant = False

        for idx, (start_page, end_page, chunk_b64) in enumerate(chunks, start=1):
            chunk_label = f"[chunk {idx}/{len(chunks)} p{start_page}-{end_page}]"
            chunk_data = self._call_claude_with_pdf_base64(chunk_b64, doc_name, chunk_label)
            if chunk_data is None:
                continue

            if chunk_data.get("relevant"):
                relevant = True
            all_casing_strings.extend(chunk_data.get("casing_strings", []))

        return {"casing_strings": all_casing_strings, "relevant": relevant}

    def extract_from_pdf(self, pdf_url: str, doc_name: str) -> dict[str, Any]:
        logger.info("Extracting document: %s", doc_name)

        has_text_layer = self.quick_text_layer_check(pdf_url)
        if has_text_layer:
            logger.info("%s: text layer detected, using text extraction path", doc_name)
            data = self._extract_with_text_path(pdf_url, doc_name)
        else:
            logger.info("%s: no text layer detected, using vision extraction path", doc_name)
            data = self._extract_with_vision_path(pdf_url, doc_name)

        self.stats["documents_processed"] += 1
        if data.get("relevant") and data.get("casing_strings"):
            self.stats["documents_returned_data"] += 1

        return data

    # Wellbore processing
    def process_wellbore(self, wellbore: str, documents: dict[str, list[dict[str, Any]]]):
        logger.info("\n%s\nProcessing wellbore: %s\n%s", "=" * 50, wellbore, "=" * 50)

        raw_strings: list[CasingString] = []
        doc_sources: list[str] = []

        for doc in documents.get("high", []):
            result = self.extract_from_pdf(doc["url"], doc["name"])
            for cs_data in result.get("casing_strings", []):
                cs = self._build_casing_string(wellbore, cs_data)
                if cs:
                    raw_strings.append(cs)
                    doc_sources.append(doc["name"])

        if self._has_critical_nulls(raw_strings):
            logger.info("%s: critical nulls detected, processing fallback docs", wellbore)
            for doc in documents.get("fallback", []):
                result = self.extract_from_pdf(doc["url"], doc["name"])
                for cs_data in result.get("casing_strings", []):
                    cs = self._build_casing_string(wellbore, cs_data)
                    if cs:
                        raw_strings.append(cs)
                        doc_sources.append(doc["name"])

        merged = self._merge_results(raw_strings, wellbore)
        unique_sources = sorted(set(doc_sources))
        for cs in merged:
            cs.source_documents = unique_sources

        self.results[wellbore] = merged
        logger.info("%s: %s casing strings found", wellbore, len(merged))

    def _build_casing_string(self, wellbore: str, data: dict[str, Any]) -> Optional[CasingString]:
        known_fields = set(CasingString.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in known_fields}
        try:
            return CasingString(wellbore=wellbore, **filtered)
        except TypeError as exc:
            logger.warning("Could not build CasingString: %s", exc)
            return None

    def _has_critical_nulls(self, casing_strings: list[CasingString]) -> bool:
        if not casing_strings:
            return True
        return any(cs.casing_type is None or cs.casing_depth_m is None for cs in casing_strings)

    # Merging and conflicts
    def _diameter_group_key(self, cs: CasingString) -> str:
        ctype = (cs.casing_type or "unknown").lower().strip()
        if cs.casing_diameter_in is not None:
            bucketed = round(cs.casing_diameter_in / DIAMETER_TOLERANCE_IN) * DIAMETER_TOLERANCE_IN
            return f"{ctype}|{bucketed}"
        return ctype

    def _merge_results(self, casing_strings: list[CasingString], wellbore: str) -> list[CasingString]:
        if not casing_strings:
            return []

        grouped: dict[str, list[CasingString]] = {}
        for cs in casing_strings:
            key = self._diameter_group_key(cs)
            grouped.setdefault(key, []).append(cs)

        numeric_fields = [
            "casing_diameter_in",
            "casing_depth_m",
            "hole_diameter_in",
            "hole_depth_m",
            "lot_fit_mud_equiv",
        ]

        merged: list[CasingString] = []
        for group_key, group in grouped.items():
            conflicts: dict[str, list[float]] = {}
            for field_name in numeric_fields:
                values = list({getattr(cs, field_name) for cs in group if getattr(cs, field_name) is not None})
                if len(values) > 1:
                    spread = max(values) - min(values)
                    if spread > DIAMETER_TOLERANCE_IN:
                        conflicts[field_name] = values

            best = max(group, key=lambda cs: sum(1 for value in asdict(cs).values() if value is not None))

            if conflicts:
                best.conflicts = conflicts
                self.stats["conflicts"].append({"wellbore": wellbore, "group": group_key, "conflicts": conflicts})
                logger.warning("Conflicts in %s / %s: %s", wellbore, group_key, conflicts)

            merged.append(best)

        merged.sort(key=lambda cs: cs.casing_depth_m or float("inf"))
        return merged

    # Output
    def output_results(self, output_path: str):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "wellbore",
            "casing_type",
            "casing_diameter_in",
            "casing_depth_m",
            "hole_diameter_in",
            "hole_depth_m",
            "lot_fit_mud_equiv",
            "formation_test_type",
            "confidence",
            "source_section",
            "source_documents",
            "conflicts",
        ]

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for wellbore in self.wellbores:
                for cs in self.results[wellbore]:
                    row = asdict(cs)
                    row["source_documents"] = "; ".join(row.get("source_documents") or [])
                    row["conflicts"] = json.dumps(row["conflicts"]) if row.get("conflicts") else ""
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

        logger.info("Results written to %s", out)
        self._log_stats()

    def _log_stats(self):
        total_tokens = self.stats["total_input_tokens"] + self.stats["total_output_tokens"]
        logger.info("\n=== Pipeline Statistics ===")
        logger.info("Total API calls:            %s", self.stats["total_api_calls"])
        logger.info(
            "Total tokens used:          %s (in: %s, out: %s)",
            total_tokens,
            self.stats["total_input_tokens"],
            self.stats["total_output_tokens"],
        )
        logger.info("Documents processed:        %s", self.stats["documents_processed"])
        logger.info("Documents returned data:    %s", self.stats["documents_returned_data"])
        logger.info("Documents failed:           %s", self.stats["documents_failed"])
        logger.info("Text-layer true:            %s", self.stats["text_layer_true"])
        logger.info("Text-layer false:           %s", self.stats["text_layer_false"])

        if self.stats["conflicts"]:
            logger.info("Value conflicts detected:   %s", len(self.stats["conflicts"]))
            for conflict in self.stats["conflicts"]:
                logger.info("  %s / %s: %s", conflict["wellbore"], conflict["group"], conflict["conflicts"])


def run_pipeline(csv_path: str, output_path: str, wellbores: Optional[list[str]] = None):
    pipeline = PipelineClean(csv_path, wellbores)
    documents = pipeline.load_csv()

    for wellbore in pipeline.wellbores:
        pipeline.process_wellbore(wellbore, documents[wellbore])

    pipeline.output_results(output_path)


def run_single_pdf(
    pdf_url: str,
    output_path: str,
    wellbore: str = "unknown",
    doc_name: str = "single_pdf",
):
    pipeline = PipelineClean(csv_path="__single_pdf_mode__.csv", wellbores=[wellbore])
    result = pipeline.extract_from_pdf(pdf_url, doc_name)

    raw_strings: list[CasingString] = []
    for cs_data in result.get("casing_strings", []):
        cs = pipeline._build_casing_string(wellbore, cs_data)
        if cs:
            raw_strings.append(cs)

    merged = pipeline._merge_results(raw_strings, wellbore)
    for cs in merged:
        cs.source_documents = [doc_name]

    pipeline.results[wellbore] = merged
    pipeline.output_results(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean wellbore casing extraction pipeline")
    parser.add_argument("--csv", default="wellbore_document_7_11.csv", help="Path to metadata CSV")
    parser.add_argument("--output", default="output/casing_strings.csv", help="Output CSV path")
    parser.add_argument("--wellbores", nargs="*", help="Wellbores to process")
    parser.add_argument("--pdf-url", help="Run single-document mode using one PDF URL")
    parser.add_argument("--doc-name", default="single_pdf", help="Document name for single-document mode")
    parser.add_argument("--single-wellbore", default="unknown", help="Wellbore label for single-document mode")
    args = parser.parse_args()

    if args.pdf_url:
        run_single_pdf(
            pdf_url=args.pdf_url,
            output_path=args.output,
            wellbore=args.single_wellbore,
            doc_name=args.doc_name,
        )
    else:
        run_pipeline(args.csv, args.output, args.wellbores)
