import json
import pymupdf  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def analyze_single_pdf(pdf_path: Path, root: Path):
    """Count pages and images in a single PDF. Returns (result, error)."""
    try:
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        img_count = sum(len(p.get_images(full=True)) for p in doc)
        doc.close()
        return {
            "filename": pdf_path.name,
            "relative_path": str(pdf_path.relative_to(root)),
            "pages": page_count,
            "images": img_count,
            "images_per_page": round(img_count / page_count, 2) if page_count else 0,
        }, None
    except Exception as e:
        return None, {
            "filename": pdf_path.name,
            "relative_path": str(pdf_path.relative_to(root)),
            "error": str(e),
        }


def analyze_pdf_folder(folder_path: str, output_json: str, max_workers: int = 12):
    folder = Path(folder_path)
    assert folder.exists(), f"Folder not found: {folder}"

    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        print("No PDFs found.")
        return

    results = []
    invalid_pdfs = []
    total_pages = total_images = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_pdf, pdf_path, folder): pdf_path for pdf_path in pdf_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Analyzing PDFs ({max_workers} workers)", unit="file"):
            result, error = future.result()
            if result:
                results.append(result)
                total_pages += result["pages"]
                total_images += result["images"]
            elif error:
                invalid_pdfs.append(error)

    valid_pdfs = len(results)
    total_pdfs = len(pdf_files)
    invalid_count = len(invalid_pdfs)

    summary = {
        "total_pdfs": total_pdfs,
        "valid_pdfs": valid_pdfs,
        "invalid_pdfs": invalid_count,
        "total_pages": total_pages,
        "total_images": total_images,
        "average_pages_per_pdf": round(total_pages / valid_pdfs, 2) if valid_pdfs else 0,
        "average_images_per_pdf": round(total_images / valid_pdfs, 2) if valid_pdfs else 0,
        "average_images_per_page": round(total_images / total_pages, 3) if total_pages else 0,
    }

    output = {"summary": summary, "valid_files": results, "invalid_files": invalid_pdfs}
    Path(output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n✅ Analysis saved to {output_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    analyze_pdf_folder("attachments", "pdf_image_analysis.json", max_workers=12)
