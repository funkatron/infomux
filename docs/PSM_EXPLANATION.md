# Tesseract PSM (Page Segmentation Mode) Explained

PSM (Page Segmentation Mode) tells Tesseract how to analyze the image layout before extracting text. Different modes work better for different types of images.

## Common PSM Modes

### PSM 6: Uniform Block of Text
- **Best for**: Documents, articles, paragraphs of text
- **What it does**: Assumes a single uniform block of text
- **Use case**: Scanned documents, PDF pages, printed text

### PSM 11: Sparse Text
- **Best for**: UI elements, scattered text, labels
- **What it does**: Looks for sparse text anywhere in the image
- **Use case**: Screenshots, UI interfaces, forms with labels

### PSM 13: Raw Line
- **Best for**: Single lines of text, code snippets
- **What it does**: No layout analysis, treats image as a single line
- **Use case**: Code screenshots, single-line text

## Why Multiple PSM Modes?

Different images have different layouts. By trying multiple PSM modes and combining results, you get:
- Better accuracy (one mode might miss text another finds)
- More complete text extraction
- Handles edge cases better

## Example from Questlog

```python
psm_modes = [6, 11, 13]
all_lines = set()

for psm in psm_modes:
    text = pytesseract.image_to_string(image, config=f'--psm {psm}')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    all_lines.update(lines)  # Combine unique lines from all modes
```

This collects text found by any of the three modes, giving more complete results.

## For Infomux

When implementing OCR, we should:
1. Try multiple PSM modes (6, 11, 13)
2. Combine results (union of all text found)
3. Apply cleanup to the combined text

This is especially important for:
- Mixed layouts (documents with sidebars)
- Screenshots of documents
- Images with both structured and unstructured text
