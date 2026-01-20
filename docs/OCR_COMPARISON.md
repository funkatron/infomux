# OCR Implementation Comparison: Questlog vs Infomux

This document compares the OCR approaches in questlog and infomux, highlighting differences and potential improvements.

## Overview

**Questlog**: Screenshot-based activity logging with OCR for UI text extraction
**Infomux**: Media pipeline with OCR for document/image text extraction

## Key Differences

### 1. OCR Engine Selection

#### Questlog
- **Primary**: EasyOCR (deep learning, better quality)
- **Fallback**: Tesseract via `pytesseract` (Python wrapper)
- **Alternative**: LLM vision models (e.g., llava) for OCR
- Uses Python libraries (`pytesseract`, `easyocr`) rather than CLI

#### Infomux (Current)
- **Planned**: Tesseract CLI only (consistent with other tools like ffmpeg/whisper-cli)
- Uses `find_tool()` pattern to locate `tesseract` binary
- No Python library dependencies for OCR

**Consideration**: Questlog uses EasyOCR as primary (better quality) with Tesseract fallback. EasyOCR uses deep learning models and provides significantly better text extraction than Tesseract, especially for:
- Handwritten text
- Complex layouts
- Low-quality images
- Non-standard fonts

**Decision needed**: Should infomux support EasyOCR (Python library) or stick with Tesseract CLI-only for consistency? EasyOCR would require adding Python dependencies but provides better quality.

### 2. OCR Configuration & PSM Modes

#### Questlog
```python
# Tries multiple PSM (Page Segmentation Mode) settings
psm_modes = [6, 11, 13]
# PSM 6 = Uniform block of text (good for code/UI)
# PSM 11 = Sparse text (good for scattered UI elements)
# PSM 13 = Raw line (no layout analysis)

# Uses character whitelist for better accuracy
whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?@#$%^&*()_+-=[]{}|\\/"\'<>~` '
config = f'--psm {psm} -c tessedit_char_whitelist={whitelist}'
```

**Key insight**: Trying multiple PSM modes and combining results improves accuracy, especially for screenshots/UI content.

#### Infomux (Current)
- Placeholder implementation
- TODO: Should adopt similar multi-PSM approach
- TODO: Consider character whitelist for document images

**Recommendation**: When implementing OCR, use multiple PSM modes (6, 11, 13) and combine results, similar to questlog.

### 3. Text Cleanup & Filtering

#### Questlog: `filter_ocr_cruft()`
**Very aggressive filtering** - designed for screenshots with UI elements:

1. **Pattern-based filtering**:
   - Menu bar items (File, Edit, View, etc.)
   - Timestamps and dates
   - System stats (GPU, CPU, memory)
   - Mostly symbols or numbers

2. **Statistical filtering**:
   - Lines with >35% symbols → skip
   - Lines with <40% letters → skip
   - Lines mostly numbers (>50% digits) → skip
   - Very short lines (<3 chars) → skip

3. **Context-aware filtering**:
   - Menu words + lots of symbols → skip
   - System overlay patterns → skip
   - Garbled text detection

**Result**: Returns list of filtered lines, not continuous text

#### Infomux: `cleanup_ocr_text()`
**Lightweight cleanup** - designed for document text:

```python
# Remove excessive whitespace
text = re.sub(r"\s+", " ", text)
# Remove line breaks that split words
text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)
# Normalize paragraph breaks
text = re.sub(r"\n{3,}", "\n\n", text)
```

**Result**: Returns continuous cleaned text string

**Key Difference**:
- Questlog filters **lines** (removes noise lines)
- Infomux cleans **text** (normalizes whitespace/structure)

### 4. Use Case Differences

#### Questlog
- **Input**: Screenshots of computer screens
- **Challenge**: Lots of UI noise (menu bars, buttons, timestamps)
- **Solution**: Aggressive line filtering
- **Output**: List of meaningful text lines

#### Infomux
- **Input**: Document images, scanned pages, photos of text
- **Challenge**: OCR artifacts (whitespace, line breaks in words)
- **Solution**: Text normalization
- **Output**: Continuous text document

### 5. Architecture Patterns

#### Questlog
- Python library approach (`pytesseract`, `easyocr`)
- Multiple OCR engines with fallback chain
- Line-by-line processing and filtering
- Integrated with vision models for context

#### Infomux
- CLI tool approach (consistent with ffmpeg/whisper-cli)
- Single OCR engine (Tesseract) for now
- Document-oriented text extraction
- Pipeline-based (extract_text → summarize)

## Recommendations for Infomux

### 1. Enhance `cleanup_ocr_text()`

Consider adding questlog-style filtering for document images:

```python
def cleanup_ocr_text(text: str, aggressive: bool = False) -> str:
    """Clean up OCR text with optional aggressive filtering."""
    # Basic cleanup (current)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if aggressive:
        # Questlog-style line filtering for noisy images
        lines = text.split('\n')
        filtered_lines = filter_ocr_cruft_lines(lines)
        text = '\n'.join(filtered_lines)

    return text.strip()
```

### 2. Multi-PSM Mode Support

When implementing `extract_text_from_image()`, use multiple PSM modes:

```python
def extract_text_from_image(image_path: Path) -> str:
    """Extract text using multiple PSM modes for better accuracy."""
    tools = get_tool_paths()
    if not tools.tesseract:
        raise StepError(...)

    all_text = set()
    psm_modes = [6, 11, 13]  # From questlog

    for psm in psm_modes:
        cmd = [
            str(tools.tesseract),
            str(image_path),
            "stdout",
            "--psm", str(psm),
            "-c", "tessedit_char_whitelist=...",  # Optional
        ]
        # Run and collect results
        # Combine unique text from all modes

    return cleanup_ocr_text(combined_text)
```

### 3. Optional Line Filtering

Add an optional step for aggressive filtering when processing screenshots or noisy images:

```python
def filter_ocr_cruft_lines(lines: list[str]) -> list[str]:
    """Filter out noise lines (adapted from questlog)."""
    # Simplified version for document images
    # Less aggressive than questlog (no UI-specific patterns)
    filtered = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        # Basic quality checks
        letters = len(re.findall(r'[a-zA-Z]', line))
        symbols = len(re.findall(r'[^\w\s]', line))
        total = len(line)
        if total > 0 and symbols / total > 0.35:
            continue
        if total > 5 and letters / total < 0.4:
            continue
        filtered.append(line)
    return filtered
```

### 4. Configuration Options

Add CLI flags for OCR behavior:

```bash
--ocr-psm-mode 6        # Single PSM mode (faster)
--ocr-aggressive        # Enable aggressive filtering
--ocr-language eng      # Language for OCR
```

## Summary

**Questlog's strengths**:
- Multi-engine OCR (EasyOCR + Tesseract + LLM)
- Sophisticated line filtering for UI noise
- Multiple PSM modes for better accuracy
- Context-aware filtering

**Infomux's strengths**:
- Consistent CLI tool architecture
- Document-oriented text extraction
- Pipeline-based processing
- Simpler, focused approach

**Best of both worlds**:
- **Consider EasyOCR**: Since it was chosen for questlog (better quality), evaluate adding EasyOCR support to infomux
- Adopt multi-PSM mode approach from questlog (for Tesseract fallback)
- Keep infomux's document-focused cleanup
- Add optional aggressive filtering for noisy images
- **Decision**: EasyOCR (better quality, Python lib) vs Tesseract CLI-only (consistency, no deps)

**Recommendation**: Support both EasyOCR (primary) and Tesseract (fallback), similar to questlog. EasyOCR provides better quality for document images, which is infomux's primary use case.
