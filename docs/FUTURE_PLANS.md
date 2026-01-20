# Future Pipeline Specifications

This document outlines planned pipelines for future implementation.

## Image Summarize Pipeline

**Name:** `image-summarize`

**Description:** Extract text from images using OCR and summarize with LLM

**Steps:**
1. `extract_text` - Extract text from image using OCR (Tesseract)
   - Input: Image file (PNG, JPG, etc.)
   - Output: `transcript.txt` (cleaned OCR text)
   - Notes:
     - OCR text is automatically cleaned via `cleanup_ocr_text()`
     - Handles common OCR artifacts (extra whitespace, line breaks in words)

2. `summarize` - Summarize extracted text with LLM
   - Input: `transcript.txt` from extract_text
   - Output: `summary.md`
   - Notes:
     - Same summarization as other text-based pipelines
     - Content type hints work (e.g., `--content-type-hint document`)

**Use Cases:**
- Screenshots of documents, articles, or notes
- Photos of whiteboards or handwritten notes (if OCR quality is sufficient)
- Scanned documents
- Images with text overlays

**Example:**
```bash
infomux run --pipeline image-summarize screenshot.png
infomux run --pipeline image-summarize --content-type-hint document scanned-doc.jpg
```

**Dependencies:**
- EasyOCR (recommended, better quality) or Tesseract OCR (fallback)
- Ollama (for summarization)

**Implementation Notes:**
- **EasyOCR recommended**: Deep learning OCR provides significantly better quality than Tesseract
  - Better for handwritten text, complex layouts, low-quality images
  - Python library (`easyocr`) - requires adding dependency
  - Model cached after first load for performance
- **Tesseract fallback**: CLI-based, consistent with existing architecture
  - Use multiple PSM modes (6, 11, 13) for better accuracy
  - Combine results from all modes
- OCR output quality varies significantly based on image quality
- May need image preprocessing (deskew, contrast enhancement) for better results
- Consider adding `--ocr-language` flag for multi-language support
- OCR cleanup step is critical - raw OCR output can be messy

---

## Document Summarize Pipeline

**Name:** `document-summarize`

**Description:** Extract text from various document formats and summarize with LLM

**Steps:**
1. `extract_text` - Extract text from document
   - Input: Document file (PDF, DOCX, images, HTML, plain text)
   - Output: `transcript.txt`
   - Notes:
     - Handles multiple formats:
       - PDF: Extract text using PDF library (e.g., PyPDF2, pdfplumber)
       - DOCX: Extract text using python-docx or similar
       - Images: Use OCR (Tesseract) - same as image-summarize
       - HTML: Extract text from HTML - same as web-summarize
       - Plain text: Use directly
     - Format detection is automatic based on file extension and content

2. `summarize` - Summarize extracted text with LLM
   - Input: `transcript.txt` from extract_text
   - Output: `summary.md`
   - Notes:
     - Same summarization as other text-based pipelines
     - Content type hints work (e.g., `--content-type-hint article`)

**Use Cases:**
- PDF articles, reports, or papers
- Word documents
- Mixed content (e.g., PDF with embedded images)
- Any document format that contains extractable text

**Example:**
```bash
infomux run --pipeline document-summarize article.pdf
infomux run --pipeline document-summarize --content-type-hint report quarterly-review.docx
infomux run --pipeline document-summarize scanned-doc.pdf  # PDF of scanned images uses OCR
```

**Dependencies:**
- Tesseract OCR (for image-based PDFs or embedded images)
- PDF extraction library (e.g., PyPDF2, pdfplumber)
- DOCX extraction library (e.g., python-docx)
- Ollama (for summarization)

**Implementation Notes:**
- PDFs can be text-based or image-based (scanned)
  - Text-based: Extract directly
  - Image-based: Use OCR on each page
- May need to handle multi-page documents
- Consider page numbers, headers/footers in extraction
- Large documents may need chunking before summarization (already handled by summarize step)

**Future Enhancements:**
- Support for more formats: RTF, ODT, EPUB
- Table extraction and preservation
- Preserve document structure (headings, sections)
- Handle encrypted/password-protected PDFs

---

## Architecture Notes

Both pipelines leverage the existing `extract_text` step, which is designed to handle multiple input formats:

1. **Format Detection:** Automatic detection based on file extension and content
2. **Unified Output:** All formats output `transcript.txt` for downstream steps
3. **Cleanup:** OCR text is automatically cleaned before summarization
4. **Reusability:** Same `summarize` step works for all text sources

This architecture allows for easy extension to new formats in the future.
