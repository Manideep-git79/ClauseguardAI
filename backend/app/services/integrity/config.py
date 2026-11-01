import os

# CLAUSEGUARD AI - PHASE 5: DOCUMENT INTEGRITY & PROMPT INJECTION CONFIGURATION

# 1. Tiny Font Size Threshold (in points)
# Standard legal text uses 10pt-12pt for body, 8pt-9pt for disclaimers, 6pt-8pt for footnotes.
# Text rendered <= 4.0pt is visually illegible without magnification and represents an anomaly.
TINY_FONT_SIZE_PT = float(os.environ.get("INTEGRITY_TINY_FONT_PT", "4.0"))

# 2. Footnote / Superscript Exemption Rules:
# Superscript symbols (flags & 1 or <= 3 chars) are legitimate typography markers.
SUPERSCRIPT_MAX_CHARS = int(os.environ.get("INTEGRITY_SUPERSCRIPT_MAX_CHARS", "3"))

# 3. Color Lightness / Contrast Threshold:
# White text on white background: RGB integer 16777215 (0xFFFFFF), or r, g, b > 240.
LIGHT_TEXT_COLOR_INT = 16777215  # 0xFFFFFF in 24-bit sRGB
RGB_LIGHTNESS_THRESHOLD = int(os.environ.get("INTEGRITY_RGB_LIGHTNESS_THRESHOLD", "240"))

# 4. Off-Page Canvas Coordinates Margin (in points):
# Margin tolerance accounts for slight bleed / crop box boundary variations.
OFF_PAGE_TOLERANCE_PT = float(os.environ.get("INTEGRITY_OFF_PAGE_TOLERANCE_PT", "15.0"))

# 5. Overlapping Text IoU Threshold:
OVERLAPPING_IOU_THRESHOLD = float(os.environ.get("INTEGRITY_OVERLAPPING_IOU", "0.85"))

# 6. Maximum Text Length to Process for Injection (prevents ReDoS / DoS):
MAX_INSPECTION_TEXT_CHARS = int(os.environ.get("INTEGRITY_MAX_TEXT_CHARS", "250000"))
