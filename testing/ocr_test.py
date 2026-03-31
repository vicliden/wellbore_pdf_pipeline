import multiprocessing as mp
# Set start method at the very top
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

import pandas as pd
import httpx
import numpy as np
import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR

# --- CONFIG ---
CSV_PATH = "wellbore_document_7_11.csv"
OUTPUT_DIR = "ocr_test_results"
PDF_CACHE = "pdf_cache"
DPI = 100
PAGES_TO_TEST = 5
CPU_WORKERS = min(4, max(cpu_count() - 1, 1))

# Global OCR model placeholder for workers
worker_ocr = None

def init_worker():
    global worker_ocr
    # Using PaddleOCR inside the worker
    worker_ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)

def process_document(args):
    # Unpack the tuple of (row_dict, url_col, name_col)
    row, url_col, name_col = args
    global worker_ocr
    
    start_time = time.time()
    name = str(row[name_col]).replace("/", "_")
    url = row[url_col]
    
    try:
        pdf_path = f"{PDF_CACHE}/{name}.pdf"
        # Download/Cache logic
        if not os.path.exists(pdf_path):
            resp = httpx.get(url, timeout=60.0)
            resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(resp.content)
            pdf_bytes = resp.content
        else:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

        images = convert_from_bytes(pdf_bytes, dpi=DPI, first_page=1, last_page=PAGES_TO_TEST)
        document_text = []

        for i, img in enumerate(images):
            # Standard PaddleOCR call
            result = worker_ocr.ocr(np.array(img), cls=False)
            
            page_words = []
            if result and result[0]:
                for line in result[0]:
                    # line[1][0] is the text string
                    page_words.append(line[1][0])
            
            page_text = " ".join(page_words)
            document_text.append(f"--- PAGE {i+1} ---\n{page_text}\n")

        output_path = f"{OUTPUT_DIR}/{name}_ocr.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE URL: {url}\n\n")
            f.write("".join(document_text))

        print(f"✅ Done: {name} ({time.time() - start_time:.1f}s)")

    except Exception as e:
        print(f"❌ FAILED: {name} | Error: {e}")

def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(PDF_CACHE).mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    
    # Identify columns once in main
    u_col = next(c for c in df.columns if "url" in c.lower())
    n_col = next(c for c in df.columns if "name" in c.lower() or "wellbore" in c.lower())

    # Prepare arguments for the pool (row, url_col_name, name_col_name)
    tasks = [(row, u_col, n_col) for row in df.to_dict("records")]

    print(f"Starting pipeline: {len(df)} docs | {CPU_WORKERS} workers")
    
    start_all = time.time()
    with Pool(processes=CPU_WORKERS, initializer=init_worker) as pool:
        pool.map(process_document, tasks)

    print(f"\n🚀 TOTAL PIPELINE TIME: {time.time() - start_all:.1f} seconds")

if __name__ == "__main__":
    main()