r"""
EndNote PDF retriever — NO external deps (PyCharm 'Run' friendly), with CHUNKED ZIPPING (<512 MB per zip).

WHAT IT DOES
1) Copies attached PDFs referenced in an EndNote XML from your .enl's .Data\PDF or .Data\Attachments folder.
2) Downloads publisher PDFs from URLs that look like PDF links ('pdf'/'pdfdirect') using urllib.request.
3) Writes a CSV log (pdf_retrieval_log.csv) in the output folder.
4) PACKS all PDFs in the output folder into multiple zip files under MAX_ZIP_MB (default 512 MB) and writes zip_manifest.csv.
   - Zips are created under <OUTPUT>\Zips\batch_001.zip, batch_002.zip, ...
   - Re-runs will SKIP files already listed in zip_manifest.csv (no duplicates).

HOW TO USE
- Put this file anywhere in your PyCharm project.
- Edit DEFAULT_XML_PATH and DEFAULT_OUTDIR below once (or pick XML via file chooser if missing).
- Press Run.

SAFE: Original PDFs remain alongside the Zips; no deletes are performed.
"""

import os
import re
import csv
import time
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen, URLError, HTTPError
import xml.etree.ElementTree as ET
import zipfile

# -------------------- CONFIG (edit these once) --------------------
DEFAULT_XML_PATH = r"C:\Users\benja\Downloads\Shirley Meng Export\ShirleyMengt2.xml"
DEFAULT_OUTDIR   = r"C:\Users\benja\Downloads\Shirley Meng Export\Export_t1"
MAX_ZIP_MB       = 100   # create zips strictly under this size
ZIP_SUBDIR_NAME  = "Zips"
# ------------------------------------------------------------------

# Optional: open file chooser if the default path is missing
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")
TIMEOUT = 30
RETRIES = 2


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_text(elem):
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()

def slugify(s: str, max_len=80):
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("\\", "-").replace("/", "-").replace(":", "-")
    s = re.sub(r"[^\w\-\.\s\(\)]", "", s)
    s = s.strip().replace(" ", "_")
    return s[:max_len] if len(s) > max_len else s

def first_author_last(author_text: str):
    if not author_text:
        return "UnknownAuthor"
    txt = author_text.strip()
    if "," in txt:
        last = txt.split(",")[0].strip()
    else:
        last = txt.split()[-1].strip()
    return slugify(last, max_len=40) or "UnknownAuthor"

def safe_filename(first_author, year, title, ext=".pdf"):
    title_short = slugify(title, max_len=64) if title else "Untitled"
    y = re.search(r"\d{4}", year or "")
    ytxt = y.group(0) if y else "n.d."
    return f"{first_author}_{ytxt}_{title_short}{ext}"

def parse_records(xml_path: Path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    return root.findall(".//record")

def find_endnote_data_dirs(records):
    data_dirs = set()
    for r in records:
        db = r.find(".//database")
        if db is not None:
            db_path = db.get("path") or read_text(db)
            if db_path:
                p = Path(db_path)
                if p.suffix.lower() == ".enl":
                    dd = p.with_suffix(".Data")
                    if dd.exists():
                        data_dirs.add(dd)
    return data_dirs

def index_all_pdfs(data_dirs):
    idx = {}
    for dd in data_dirs:
        for root in [dd / "PDF", dd / "Attachments"]:
            if root.exists():
                for path in root.rglob("*.pdf"):
                    idx.setdefault(path.name.lower(), []).append(path)
    return idx

def try_copy_internal(internal_hint: str, filename_index: dict, out_dir: Path):
    m = re.search(r"internal-pdf://[^/]+/(.+)$", internal_hint.strip())
    if not m:
        return None, "NoMatch"
    hint_name = m.group(1).strip()
    candidates = filename_index.get(hint_name.lower())
    if not candidates:
        return None, "NotIndexed"
    src = candidates[0]
    dst = out_dir / hint_name
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        for i in range(1, 5000):
            alt = out_dir / f"{stem}__dup{i}{suf}"
            if not alt.exists():
                dst = alt
                break
    shutil.copy2(src, dst)
    return dst, "Copied"

def pick_external_pdf_urls(url_nodes):
    urls = []
    for u in url_nodes:
        url = read_text(u)
        if not url:
            continue
        if "pdf" in url.lower() or "pdfdirect" in url.lower():
            urls.append(url)
    return urls

def best_guess_filename(first_author, year, title, url):
    if first_author or year or title:
        return safe_filename(first_author or "Unknown", year or "", title or "", ".pdf")
    name = os.path.basename(urlparse(url).path) or "download.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name

def urlretrieve_with_retries(url, dest_path: Path):
    last_exc = None
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, RETRIES + 2):
        try:
            with urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
                if dest_path.suffix.lower() != ".pdf":
                    dest_path = dest_path.with_suffix(".pdf")
                with open(dest_path, "wb") as f:
                    f.write(data)
            return dest_path, "Downloaded"
        except (HTTPError, URLError, TimeoutError, Exception) as e:
            last_exc = e
            time.sleep(1.2 * attempt)
    return None, f"DownloadFailed: {last_exc}"

# -------------------- ZIPPING (chunked) --------------------

def load_zipped_manifest(zip_dir: Path):
    """Return a set of absolute file paths already recorded in manifest to avoid re-zipping duplicates."""
    manifest_path = zip_dir / "zip_manifest.csv"
    already = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("FileAbsPath"):
                    already.add(row["FileAbsPath"])
    return already

def append_manifest(zip_dir: Path, batch_name: str, files):
    manifest_path = zip_dir / "zip_manifest.csv"
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["BatchZip", "FileRelPath", "FileAbsPath", "SizeBytes"])
        for p in files:
            writer.writerow([batch_name, str(p.name), str(p.resolve()), p.stat().st_size])

def chunk_and_zip_pdfs(out_dir: Path, max_zip_mb: int = 512, zip_subdir: str = "Zips"):
    max_bytes = max_zip_mb * 1024 * 1024
    zip_dir = ensure_dir(out_dir / zip_subdir)

    # Collect all PDFs in out_dir (not recursing into Zips) and skip existing zips
    all_pdfs = [p for p in out_dir.glob("*.pdf") if p.is_file()]
    if not all_pdfs:
        print("[ZIP] No PDFs found to zip. Skipping.")
        return

    already = load_zipped_manifest(zip_dir)
    to_zip = [p for p in all_pdfs if str(p.resolve()) not in already]
    if not to_zip:
        print("[ZIP] Nothing new to zip (everything already in manifest).")
        return

    # Sort by size descending so single large files get their own batch first if needed
    to_zip.sort(key=lambda p: p.stat().st_size, reverse=True)

    batches = []
    current_batch, current_size = [], 0
    overhead_per_file = 200  # conservative per-file zip overhead

    for p in to_zip:
        fsz = p.stat().st_size + overhead_per_file
        if fsz > max_bytes:
            # Edge case: one file exceeds limit; put it alone (will be slightly > limit if ZIP_STORED header exceeds)
            print(f"[ZIP][WARN] Single file exceeds {max_zip_mb} MB: {p.name} ({fsz/1024/1024:.1f} MB). "
                  f"Creating a dedicated zip for it.")
            batches.append([p])
            continue

        if current_size + fsz > max_bytes and current_batch:
            batches.append(current_batch)
            current_batch, current_size = [], 0

        current_batch.append(p)
        current_size += fsz

    if current_batch:
        batches.append(current_batch)

    # Create batch_###.zip files
    existing_batches = list(zip_dir.glob("batch_*.zip"))
    start_idx = 1
    if existing_batches:
        # Continue numbering
        nums = []
        for z in existing_batches:
            m = re.search(r"batch_(\d+)\.zip$", z.name)
            if m:
                nums.append(int(m.group(1)))
        if nums:
            start_idx = max(nums) + 1

    for i, files in enumerate(batches, start=start_idx):
        zip_name = f"batch_{i:03d}.zip"
        zip_path = zip_dir / zip_name
        print(f"[ZIP] Writing {zip_name} with {len(files)} files…")

        # Use STORED to make size predictable (PDFs are already compressed)
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for p in files:
                zf.write(p, arcname=p.name)
        # Quick sanity check
        zsize = zip_path.stat().st_size
        if zsize > max_bytes:
            print(f"[ZIP][WARN] {zip_name} is {zsize/1024/1024:.1f} MB (> {max_zip_mb} MB). "
                  f"This can happen with zip overhead; consider lowering MAX_ZIP_MB slightly and re-run.")
        append_manifest(zip_dir, zip_name, files)

    print(f"[ZIP][DONE] Created {len(batches)} zip(s) in: {zip_dir}")


# -------------------- MAIN RETRIEVAL --------------------

def run_retrieval(xml_path: Path, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    log_path = out_dir / "pdf_retrieval_log.csv"

    print(f"[INFO] Parsing XML: {xml_path}")
    records = parse_records(xml_path)

    data_dirs = find_endnote_data_dirs(records)
    filename_index = index_all_pdfs(data_dirs)

    print(f"[INFO] Found {len(records)} records; indexed PDFs in {len(data_dirs)} EndNote .Data folder(s).")
    if not filename_index:
        print("[WARN] No internal PDFs found in .Data. Will try external links only.")

    with open(log_path, "w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        writer.writerow([
            "RecNumber", "Title", "Year", "FirstAuthor", "DOI",
            "Source", "URL_or_Path", "SavedAs", "Status", "Notes"
        ])

        for rec in records:
            recnum = read_text(rec.find(".//rec-number"))
            title  = read_text(rec.find(".//titles/title"))
            year   = read_text(rec.find(".//dates/year")) or read_text(rec.find(".//year"))
            doi    = read_text(rec.find(".//electronic-resource-num")) or ""

            first_author_node = rec.find(".//contributors/authors/author")
            first_author = first_author_last(read_text(first_author_node)) if first_author_node is not None else "UnknownAuthor"

            internal_urls = [read_text(u) for u in rec.findall(".//pdf-urls/url")]
            external_urls = pick_external_pdf_urls(rec.findall(".//urls//url"))

            saved_any = False

            # Try internal copies first
            for iurl in internal_urls:
                if not iurl.lower().startswith("internal-pdf://"):
                    continue
                try:
                    copied_path, status = try_copy_internal(iurl, filename_index, out_dir)
                    if copied_path:
                        nice_name = best_guess_filename(first_author, year, title, str(copied_path))
                        nice_path = out_dir / nice_name
                        if not nice_path.exists():
                            try:
                                shutil.copy2(copied_path, nice_path)
                                writer.writerow([recnum, title, year, first_author, doi,
                                                 "internal", str(copied_path), str(nice_path), "Copied", ""])
                            except Exception as e:
                                writer.writerow([recnum, title, year, first_author, doi,
                                                 "internal", str(copied_path), "", "Copied (no nice dup)", f"{e}"])
                        else:
                            writer.writerow([recnum, title, year, first_author, doi,
                                             "internal", str(copied_path), str(nice_path), "Exists", ""])
                        saved_any = True
                    else:
                        writer.writerow([recnum, title, year, first_author, doi,
                                         "internal", iurl, "", status, ""])
                except Exception as e:
                    writer.writerow([recnum, title, year, first_author, doi,
                                     "internal", iurl, "", "Error", f"{e}"])

            # If internal not found, try external downloads
            if not saved_any:
                for eurl in external_urls:
                    try:
                        fname = best_guess_filename(first_author, year, title, eurl)
                        dest = out_dir / fname
                        if dest.exists() and dest.stat().st_size > 5_000:
                            writer.writerow([recnum, title, year, first_author, doi,
                                             "external", eurl, str(dest), "Exists", ""])
                            saved_any = True
                            break
                        saved_path, status = urlretrieve_with_retries(eurl, dest)
                        if saved_path:
                            writer.writerow([recnum, title, year, first_author, doi,
                                             "external", eurl, str(saved_path), status, ""])
                            saved_any = True
                            break
                        else:
                            writer.writerow([recnum, title, year, first_author, doi,
                                             "external", eurl, "", status, ""])
                    except Exception as e:
                        writer.writerow([recnum, title, year, first_author, doi,
                                         "external", eurl, "", "Error", f"{e}"])

            if not saved_any and not internal_urls and not external_urls:
                writer.writerow([recnum, title, year, first_author, doi,
                                 "none", "", "", "NoLinks", "No internal or external PDF links in XML"])

    print(f"[DONE] Saved PDFs (and log) to: {out_dir}")
    print(f"[INFO] Log: {log_path}")


def main():
    xml_path = Path(DEFAULT_XML_PATH)
    out_dir  = Path(DEFAULT_OUTDIR)

    # Pick XML interactively if missing
    if not xml_path.exists():
        print(f"[WARN] XML not found at default path:\n  {xml_path}")
        if TK_AVAILABLE:
            print("[INFO] Opening file picker…")
            root = tk.Tk(); root.withdraw()
            chosen = filedialog.askopenfilename(
                title="Select EndNote XML",
                filetypes=[("EndNote XML", "*.xml"), ("All files", "*.*")]
            )
            if not chosen:
                print("[ABORT] No file selected.")
                return
            xml_path = Path(chosen)
        else:
            print("[ABORT] Tkinter not available and default XML missing. Please edit DEFAULT_XML_PATH.")
            return

    ensure_dir(out_dir)

    # 1) Retrieve PDFs
    run_retrieval(xml_path, out_dir)

    # 2) Chunked zipping (< MAX_ZIP_MB)
    chunk_and_zip_pdfs(out_dir, max_zip_mb=MAX_ZIP_MB, zip_subdir=ZIP_SUBDIR_NAME)


if __name__ == "__main__":
    main()
