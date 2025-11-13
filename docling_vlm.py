import base64
import io
import logging
from pathlib import Path
import os

import requests
from PIL import Image
from pdf2image import convert_from_path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import PictureItem, DescriptionMetaField

from dotenv import load_dotenv

load_dotenv()

# Find a correct meta class by introspection
try:
    # Create a dummy picture and get its meta type
    _tmp_pic = PictureItem()
    MetaClass = type(_tmp_pic.meta)
except Exception:
    print("Warning: Could not determine MetaClass.")
    exit(1)

# ---------------- Provider Configuration ----------------

PROVIDERS = {
    "ollama": {
        "url": "http://localhost:11434/api/chat",
        "model": "qwen3-vl:32b",
        "auth": None,  # no API key needed
        "format": "ollama",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "auth": lambda: {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
        "format": "openai",
    },
    "gemini": {
        "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={os.getenv('GEMINI_API_KEY', '')}",
        "model": "gemini-2.5-flash",
        "auth": None,  # key is embedded in URL
        "format": "gemini",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-4-maverick:free",  # or gemini-2.0-flash, mistral-large, etc.
        "auth": lambda: {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}"},
        "format": "openai",  # same format as OpenAI’s API
    },
}


# Select provider here (or from environment variable)
PROVIDER = os.getenv("VISION_BACKEND", "ollama")
CFG = PROVIDERS[PROVIDER]
PROMPT = "If an image of a magazine: return empty string. If a diagram: represent in table form. If text: just repeat the text as if doing OCR, unless the text is incoherent. If none of the above, return empty string. No descriptions. No framing."

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("docling_ollama")


def describe_pil(img: Image.Image, prompt: str) -> str:
    """Send a PIL image to the configured multimodal backend (Ollama / OpenAI / Gemini / Openrouter)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # --- Ollama ---
    if CFG["format"] == "ollama":
        payload = {
            "model": CFG["model"],
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        url = CFG["url"]

    # --- OpenAI ---
    elif CFG["format"] == "openai":
        payload = {
            "model": CFG["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if CFG["auth"]:
            headers.update(CFG["auth"]())
        url = CFG["url"]

    # --- OpenRouter ---
    elif CFG["format"] == "openrouter":
        payload = {
            "model": CFG["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if CFG["auth"]:
            headers.update(CFG["auth"]())
        url = CFG["url"]

    # --- Gemini ---
    elif CFG["format"] == "gemini":
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64}},
                    ],
                }
            ]
        }
        headers = {"Content-Type": "application/json"}
        url = CFG["url"]

    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")

    LOGGER.debug(f"Calling {PROVIDER} at {url}")
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        LOGGER.error("Error %s: %s", r.status_code, r.text[:500])
        raise

    data = r.json()

    # --- Response parsing ---
    if CFG["format"] == "ollama":
        return (data.get("message", {}) or {}).get("content", "").strip()
    elif CFG["format"] == "openai":
        return data["choices"][0]["message"]["content"].strip()
    elif CFG["format"] == "gemini":
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

# ---------------- Coordinate helper ----------------

def pdf_to_pixel_bbox(bbox, page_width_px, page_height_px, pdf_width_pt=595.0, pdf_height_pt=842.0):
    """Convert Docling bbox (bottom-left origin, PDF points) to pixel coordinates (top-left origin)."""
    scale_x = page_width_px / pdf_width_pt
    scale_y = page_height_px / pdf_height_pt

    l = int(bbox.l * scale_x)
    r = int(bbox.r * scale_x)
    # flip Y because PDF origin is bottom-left
    t = page_height_px - int(bbox.t * scale_y)
    b = page_height_px - int(bbox.b * scale_y)

    # Ensure correct order
    if t > b:
        t, b = b, t
    return (l, t, r, b)

# ---------------- Main converter ----------------

def convert_pdf_to_md_with_local_ollama(pdf_path: str, output_md: str, dpi: int = 72):
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"Missing PDF: {pdf_path}"

    # Convert PDF with Docling (skip picture desc as we use VLM)
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False
    pipeline_opts.do_picture_description = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )

    LOGGER.info("Converting PDF: %s", pdf_path)
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Render all pages for cropping
    LOGGER.info("Rendering pages at %d DPI…", dpi)
    pages = convert_from_path(str(pdf_path), dpi=dpi)

    total_pics, described = 0, 0

    # Iterate over all detected pictures
    for item, _ in doc.iterate_items(traverse_pictures=True, with_groups=True):
        if not isinstance(item, PictureItem):
            continue
        total_pics += 1

        if not getattr(item, "prov", None):
            LOGGER.warning("Picture without prov info; skipping.")
            continue

        prov = item.prov[0]
        page_no = getattr(prov, "page_no", None)
        bbox = getattr(prov, "bbox", None)
        if page_no is None or bbox is None:
            LOGGER.warning("Picture missing page_no or bbox; skipping.")
            continue

        page_idx = page_no - 1
        if page_idx < 0 or page_idx >= len(pages):
            LOGGER.warning("Page index %d out of range; skipping.", page_idx)
            continue

        page_img = pages[page_idx]
        pw, ph = page_img.size
        crop_box = pdf_to_pixel_bbox(bbox, pw, ph)
        l, t, r, b = crop_box

        if r - l < 10 or b - t < 10:
            LOGGER.warning("Tiny or invalid bbox on page %d; skipping.", page_no)
            continue

        pil_img = page_img.crop(crop_box)

        try:
            caption = describe_pil(pil_img, prompt=PROMPT)
            if caption:
                if not getattr(item, "meta", None) or not hasattr(item.meta, "get_custom_part"):
                    item.meta = MetaClass()
                item.meta.description = DescriptionMetaField(text=caption)
                described += 1
                LOGGER.info("Page %d: %s", page_no, caption)


            else:
                LOGGER.warning("Empty caption for picture on page %d.", page_no)
        except Exception as e:
            LOGGER.warning("Captioning failed on page %d: %s", page_no, e)

    LOGGER.info("Pictures found: %d; described: %d", total_pics, described)

    # Export Markdown with descriptions
    md = doc.export_to_markdown(
        include_annotations=False,
        use_legacy_annotations=False,
        allowed_meta_names={"description"},
        mark_meta=False,
    )
    Path(output_md).write_text(md, encoding="utf-8")
    LOGGER.info("✅ Markdown saved to %s", output_md)


if __name__ == "__main__":
    convert_pdf_to_md_with_local_ollama(
        pdf_path="readly-report-10pages-pdf/abf182b2-b649-51b7-81c1-0e140e7cbc3d_1-pages-1.pdf",
        output_md="output.md",
        dpi=72,
    )
