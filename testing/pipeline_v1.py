"""
Wellbore PDF Pipeline
Extract casing string data from PDFs and output structured table.
"""

import csv
import io
import json
import logging
import base64
import time
import httpx
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import anthropic
import pypdf
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

error_handler = logging.FileHandler("errors.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(error_handler)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_PAGES_PER_CHUNK   = 50   # stay under the 100-page API limit
SECONDS_BETWEEN_CALLS = 5    # conservative rate-limit guard (30k tokens/min)
DIAMETER_TOLERANCE_IN = 0.5  # bucket size for grouping casing strings by diameter

# ── Schema ────────────────────────────────────────────────────────────────────

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
    source_documents: list = field(default_factory=list)
    conflicts: Optional[dict] = None


# ── Priority classification ───────────────────────────────────────────────────

HIGH_PRIORITY = [
    "WELL_COMPLETION_REPORT", "WDSS_GENERAL_INFORMATION", "WDSS_COMPLETION_LOG",
    "INDIVIDUAL_WELL_RECORD", "DRILLING_MUD_RECORD", "DRILLING_MUD_REPORT",
    "DRILLING_FLUID_SUMMARY", "DRILL_STEM_TEST_DATA", "FORMATION_TEST",
    "EXPLORATORY_TEST",
]

FALLBACK = [
    "WELL_PRODUCTION_TEST", "TEMPORARY_SUSPENSION_PROCEDURE", "CHANGE_IN_DRILLING_PROGRAM",
]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in Norwegian Continental Shelf (NCS) well documentation.
Your task is to extract casing string data from oil well PDFs from the SODIR database.

Casing types: conductor, surface, intermediate, production, liner
Typical casing diameters: 30", 20", 13 3/8", 9 5/8", 7"
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
- medium: value inferred or units had to be converted
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


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, csv_path: str, wellbores: list = None):
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
        }

    # ── CSV loading ───────────────────────────────────────────────────────────

    def load_csv(self) -> dict:
        """Load metadata CSV and group documents by wellbore and priority."""
        grouped = {wb: {"high": [], "fallback": [], "skip": []} for wb in self.wellbores}

        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                wellbore = row.get("wlbName") or row.get("wellbore") or row.get("Wellbore")
                doc_name = row.get("wlbDocumentName") or row.get("documentName") or row.get("name", "")
                pdf_url  = row.get("wlbDocumentUrl")  or row.get("url")

                if not wellbore or not pdf_url:
                    continue
                if wellbore not in grouped:
                    continue

                priority = self._classify_priority(doc_name)
                grouped[wellbore][priority].append({"name": doc_name, "url": pdf_url})

        for wb, groups in grouped.items():
            logger.info(
                f"{wb}: {len(groups['high'])} high priority, "
                f"{len(groups['fallback'])} fallback, "
                f"{len(groups['skip'])} skipped"
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

    # ── PDF fetching & chunking ───────────────────────────────────────────────

    import io
    import logging
    from typing import Optional

    import httpx
    import pypdf


    logger = logging.getLogger(__name__)


    def fetch_pdf_head(self, url: str, byte_limit: int = 65536) -> Optional[bytes]:
        """
        Download only the first portion of a PDF using HTTP Range requests.

        Default: first 64 KB (usually enough to parse page 1 objects).
        """
        headers = {"Range": f"bytes=0-{byte_limit - 1}"}

        try:
            response = httpx.get(
                url,
                headers=headers,
                timeout=30,
                follow_redirects=True,
            )

            if response.status_code in (200, 206):
                return response.content

            logger.warning(
                f"Server ignored range request for {url} "
                f"(status={response.status_code})"
            )
            return None

        except Exception as e:
            logger.error(f"Partial PDF fetch failed for {url}: {e}")
            return None


    def has_text_layer_partial(self, pdf_head_bytes: bytes) -> Optional[bool]:
        """
        Attempt to detect whether a PDF has a text layer using only partial bytes.

        Returns:
            True  → text detected
            False → no text detected
            None  → inconclusive (PDF structure incomplete in partial download)
        """
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_head_bytes))

            if not reader.pages:
                return None

            first_page = reader.pages[0]

            text = first_page.extract_text()

            if text and text.strip():
                return True

            return False

        except Exception:
            # Partial PDFs sometimes lack xref tables → parsing fails
            return None


    def quick_text_layer_check(self, url: str) -> bool:
        """
        Lightweight check whether a PDF likely has extractable text.

        Strategy:
            1. download first 64 KB
            2. attempt page-1 text extraction
            3. fallback assumption = unknown → treat as scanned
        """
        head_bytes = self.fetch_pdf_head(url)

        if not head_bytes:
            return False

        result = self.has_text_layer_partial(head_bytes)

        if result is None:
            logger.info("Partial parse inconclusive — assuming scanned PDF")
            return False

        return result

    def _extract_text_from_pdf(self, pdf_bytes: bytes, max_pages: int = None) -> Optional[str]:
        """
        Extract text from PDF using pypdf.
        Returns concatenated text from all pages (or first max_pages) or None on error.
        """
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            all_text = []
            pages_to_process = reader.pages if max_pages is None else reader.pages[:max_pages]
            for page in pages_to_process:
                text = page.extract_text()
                if text:
                    all_text.append(text)
            return "\n".join(all_text) if all_text else None
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return None

    def _split_pdf_chunks(self, pdf_bytes: bytes) -> list[str]:
        """
        Split a PDF into base64-encoded chunks of MAX_PAGES_PER_CHUNK pages.
        Returns a list of base64 strings, one per chunk.
        """
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)

        if total_pages <= MAX_PAGES_PER_CHUNK:
            return [base64.standard_b64encode(pdf_bytes).decode("utf-8")]

        logger.info(f"PDF has {total_pages} pages - splitting into chunks of {MAX_PAGES_PER_CHUNK}")
        chunks = []
        for start in range(0, total_pages, MAX_PAGES_PER_CHUNK):
            end = min(start + MAX_PAGES_PER_CHUNK, total_pages)
            writer = pypdf.PdfWriter()
            for page_num in range(start, end):
                writer.add_page(reader.pages[page_num])
            buf = io.BytesIO()
            writer.write(buf)
            chunks.append(base64.standard_b64encode(buf.getvalue()).decode("utf-8"))
            logger.info(f"  Chunk {len(chunks)}: pages {start + 1}-{end}")

        return chunks

    # ── Claude API call ───────────────────────────────────────────────────────

    def _call_api(self, pdf_b64: str, doc_name: str, chunk_info: str = "", model: str = "claude-opus-4-5") -> Optional[dict]:
        """
        Send one PDF chunk to Claude. Sleeps before each call to respect
        the rate limit, and retries up to 3 times on 429 errors.
        Returns parsed dict or None on unrecoverable error.
        """
        label = f"{doc_name}{chunk_info}"
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
                self.stats["total_input_tokens"]  += message.usage.input_tokens
                self.stats["total_output_tokens"] += message.usage.output_tokens

                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

                return json.loads(raw)

            except anthropic.RateLimitError as e:
                wait = 60
                logger.warning(
                    f"Rate limited on {label} (attempt {attempt+1}/{max_retries}), "
                    f"waiting {wait}s: {e}"
                )
                time.sleep(wait)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for {label}: {e}")
                return None

            except anthropic.BadRequestError as e:
                logger.error(f"Bad request for {label} (unrecoverable): {e}")
                return None

            except Exception as e:
                logger.error(f"API error for {label} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)

        logger.error(f"All retries exhausted for {label}")
        return None

    # ── Document extraction ───────────────────────────────────────────────────

    def _call_api_with_text(self, text: str, doc_name: str) -> Optional[dict]:
        """
        Send extracted text to Claude for casing data extraction.
        Uses better model for processing text-extracted content.
        """
        label = f"{doc_name} [text extraction]"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                time.sleep(SECONDS_BETWEEN_CALLS)

                message = self.client.messages.create(
                    model="claude-opus-4-5",
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
                self.stats["total_input_tokens"]  += message.usage.input_tokens
                self.stats["total_output_tokens"] += message.usage.output_tokens

                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

                return json.loads(raw)

            except anthropic.RateLimitError as e:
                wait = 60
                logger.warning(
                    f"Rate limited on {label} (attempt {attempt+1}/{max_retries}), "
                    f"waiting {wait}s: {e}"
                )
                time.sleep(wait)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for {label}: {e}")
                return None

            except anthropic.BadRequestError as e:
                logger.error(f"Bad request for {label} (unrecoverable): {e}")
                return None

            except Exception as e:
                logger.error(f"API error for {label} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)

        logger.error(f"All retries exhausted for {label}")
        return None

    def extract_from_pdf(self, pdf_url: str, doc_name: str) -> dict:
        """Download PDF, check for text layer, extract using appropriate method."""
        logger.info(f"Extracting: {doc_name}")

        pdf_bytes = self._fetch_pdf_bytes(pdf_url)
        if not pdf_bytes:
            self.stats["documents_failed"] += 1
            return {"casing_strings": [], "relevant": False}

        # Check if PDF has text layers
        has_text = self._has_text_layer(pdf_bytes)
        
        if has_text:
            logger.info(f"{doc_name}: PDF has text layer, extracting text")
            extracted_text = self._extract_text_from_pdf(pdf_bytes)
            if extracted_text:
                # Use low-cost model to extract text
                data = self._call_api_text_extraction(extracted_text, doc_name)
                if data:
                    # Use better model to process text
                    refined_data = self._call_api_with_text(extracted_text, doc_name)
                    if refined_data:
                        data = refined_data
                
                self.stats["documents_processed"] += 1
                if data and data.get("relevant") and data.get("casing_strings"):
                    self.stats["documents_returned_data"] += 1
                return data if data else {"casing_strings": [], "relevant": False}
            else:
                logger.warning(f"{doc_name}: Failed to extract text, falling back to PDF processing")
        
        # Use low-cost model for PDF (image-only or text extraction failed)
        logger.info(f"{doc_name}: Using low-cost model for PDF processing")
        try:
            chunks = self._split_pdf_chunks(pdf_bytes)
        except Exception as e:
            logger.error(f"Could not split PDF {doc_name}: {e}")
            self.stats["documents_failed"] += 1
            return {"casing_strings": [], "relevant": False}

        all_casing_strings = []
        relevant = False

        for i, chunk_b64 in enumerate(chunks):
            chunk_info = f" [chunk {i+1}/{len(chunks)}]" if len(chunks) > 1 else ""
            # Use low-cost model first
            data = self._call_api(chunk_b64, doc_name, chunk_info, model="claude-haiku-4-5")

            if data is None:
                self.stats["documents_failed"] += 1
                continue

            if data.get("relevant"):
                relevant = True
            all_casing_strings.extend(data.get("casing_strings", []))

        self.stats["documents_processed"] += 1
        if relevant and all_casing_strings:
            self.stats["documents_returned_data"] += 1

        return {"casing_strings": all_casing_strings, "relevant": relevant}
    
    def _call_api_text_extraction(self, text: str, doc_name: str) -> Optional[dict]:
        """
        Use low-cost model to extract casing data from text.
        """
        label = f"{doc_name} [text extraction - low cost]"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                time.sleep(SECONDS_BETWEEN_CALLS)

                message = self.client.messages.create(
                    model="claude-haiku-4-5",
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
                self.stats["total_input_tokens"]  += message.usage.input_tokens
                self.stats["total_output_tokens"] += message.usage.output_tokens

                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

                return json.loads(raw)

            except anthropic.RateLimitError as e:
                wait = 60
                logger.warning(
                    f"Rate limited on {label} (attempt {attempt+1}/{max_retries}), "
                    f"waiting {wait}s: {e}"
                )
                time.sleep(wait)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for {label}: {e}")
                return None

            except anthropic.BadRequestError as e:
                logger.error(f"Bad request for {label} (unrecoverable): {e}")
                return None

            except Exception as e:
                logger.error(f"API error for {label} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)

        logger.error(f"All retries exhausted for {label}")
        return None

    # ── Wellbore processing ───────────────────────────────────────────────────

    def process_wellbore(self, wellbore: str, documents: dict):
        logger.info(f"\n{'='*50}\nProcessing wellbore: {wellbore}\n{'='*50}")
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
            logger.info(f"{wellbore}: critical nulls found, processing fallback documents")
            for doc in documents.get("fallback", []):
                result = self.extract_from_pdf(doc["url"], doc["name"])
                for cs_data in result.get("casing_strings", []):
                    cs = self._build_casing_string(wellbore, cs_data)
                    if cs:
                        raw_strings.append(cs)
                        doc_sources.append(doc["name"])

        merged = self._merge_results(raw_strings, wellbore)
        unique_sources = list(set(doc_sources))
        for cs in merged:
            cs.source_documents = unique_sources

        self.results[wellbore] = merged
        logger.info(f"{wellbore}: {len(merged)} casing strings found")

    def _build_casing_string(self, wellbore: str, data: dict) -> Optional[CasingString]:
        """Build a CasingString from API response dict, ignoring unknown keys."""
        known_fields = set(CasingString.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in known_fields}
        try:
            return CasingString(wellbore=wellbore, **filtered)
        except TypeError as e:
            logger.warning(f"Could not build CasingString: {e}")
            return None

    def _has_critical_nulls(self, casing_strings: list[CasingString]) -> bool:
        if not casing_strings:
            return True
        return any(
            cs.casing_type is None or cs.casing_depth_m is None
            for cs in casing_strings
        )

    # ── Merging ───────────────────────────────────────────────────────────────

    def _diameter_group_key(self, cs: CasingString) -> str:
        """
        Group key combining casing_type with a bucketed diameter.
        Prevents distinct strings (e.g. 13 3/8" and 9 5/8" intermediate)
        from being collapsed into the same group just because they share a type label.
        """
        ctype = (cs.casing_type or "unknown").lower().strip()

        if cs.casing_diameter_in is not None:
            bucketed = round(cs.casing_diameter_in / DIAMETER_TOLERANCE_IN) * DIAMETER_TOLERANCE_IN
            return f"{ctype}|{bucketed}"

        return ctype  # fallback when diameter unknown

    def _merge_results(self, casing_strings: list[CasingString], wellbore: str) -> list[CasingString]:
        """
        Merge duplicate casing strings, detect conflicts, keep best record.
        Groups by (casing_type, diameter bucket) to avoid collapsing
        distinct strings that share only their type label.
        """
        if not casing_strings:
            return []

        grouped: dict[str, list[CasingString]] = {}
        for cs in casing_strings:
            key = self._diameter_group_key(cs)
            grouped.setdefault(key, []).append(cs)

        numeric_fields = [
            "casing_diameter_in", "casing_depth_m",
            "hole_diameter_in", "hole_depth_m", "lot_fit_mud_equiv",
        ]
        merged = []

        for group_key, group in grouped.items():
            conflicts = {}
            for f in numeric_fields:
                values = list({getattr(cs, f) for cs in group if getattr(cs, f) is not None})
                if len(values) > 1:
                    spread = max(values) - min(values)
                    if spread > DIAMETER_TOLERANCE_IN:
                        conflicts[f] = values

            best = max(
                group,
                key=lambda cs: sum(1 for v in asdict(cs).values() if v is not None),
            )

            if conflicts:
                best.conflicts = conflicts
                self.stats["conflicts"].append(
                    {"wellbore": wellbore, "group": group_key, "conflicts": conflicts}
                )
                logger.warning(f"Conflicts in {wellbore} / {group_key}: {conflicts}")

            merged.append(best)

        # Sort by casing depth ascending for readability
        merged.sort(key=lambda cs: cs.casing_depth_m or float("inf"))
        return merged

    # ── Output ────────────────────────────────────────────────────────────────

    def output_results(self, output_path: str):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "wellbore", "casing_type", "casing_diameter_in", "casing_depth_m",
            "hole_diameter_in", "hole_depth_m", "lot_fit_mud_equiv", "formation_test_type",
            "confidence", "source_section", "source_documents", "conflicts",
        ]

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for wellbore in self.wellbores:
                for cs in self.results[wellbore]:
                    row = asdict(cs)
                    row["source_documents"] = "; ".join(row.get("source_documents") or [])
                    row["conflicts"] = json.dumps(row["conflicts"]) if row["conflicts"] else ""
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

        logger.info(f"Results written to {out}")
        self._log_stats()

    def _log_stats(self):
        total_tokens = self.stats["total_input_tokens"] + self.stats["total_output_tokens"]
        logger.info("\n=== Pipeline Statistics ===")
        logger.info(f"Total API calls:            {self.stats['total_api_calls']}")
        logger.info(f"Total tokens used:          {total_tokens} "
                    f"(in: {self.stats['total_input_tokens']}, out: {self.stats['total_output_tokens']})")
        logger.info(f"Documents processed:        {self.stats['documents_processed']}")
        logger.info(f"Documents returned data:    {self.stats['documents_returned_data']}")
        logger.info(f"Documents failed:           {self.stats['documents_failed']}")
        if self.stats["conflicts"]:
            logger.info(f"Value conflicts detected:   {len(self.stats['conflicts'])}")
            for c in self.stats["conflicts"]:
                logger.info(f"  {c['wellbore']} / {c['group']}: {c['conflicts']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_pipeline(csv_path: str, output_path: str, wellbores: list = None):
    pipeline = Pipeline(csv_path, wellbores)
    documents = pipeline.load_csv()
    for wellbore in pipeline.wellbores:
        pipeline.process_wellbore(wellbore, documents[wellbore])
    pipeline.output_results(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wellbore casing extraction pipeline")
    parser.add_argument("--csv",       default="wellbore_document_7_11.csv", help="Path to metadata CSV")
    parser.add_argument("--output",    default="output/casing_strings.csv",  help="Output CSV path")
    parser.add_argument("--wellbores", nargs="*", help="Wellbores to process (default: all four)")
    args = parser.parse_args()

    run_pipeline(args.csv, args.output, args.wellbores)