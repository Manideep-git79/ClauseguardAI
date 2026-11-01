import re
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "png", "jpg", "jpeg"}
ALLOWED_DOCUMENT_TYPES = {"rental_agreement", "job_offer_letter", "insurance_policy"}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_PAGE_COUNT = 500

class FileValidationError(Exception):
    """Base exception for file validation failures."""
    pass

class EncryptedFileError(FileValidationError):
    """Raised when an uploaded document is password-protected or encrypted."""
    pass

class CorruptedFileError(FileValidationError):
    """Raised when an uploaded document is malformed, truncated, or unreadable."""
    pass

class EmptyFileError(FileValidationError):
    """Raised when an uploaded document contains 0 bytes or zero pages."""
    pass

class FileLimitExceededError(FileValidationError):
    """Raised when a document exceeds maximum allowed size or page count."""
    pass

PROMPT_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"(?i)\byou\s+are\s+now\b",
    r"(?i)\breveal\s+(your\s+)?(system\s+prompt|instructions)\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior)\s+rules\b",
    r"(?i)\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|developer)\b",
    r"(?i)\bpretend\s+you\s+are\b",
    r"(?i)\bsystem\s*:\s*",
    r"(?i)\bDAN\s+mode\b",
]

def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def validate_document_type(doc_type: str) -> bool:
    return doc_type in ALLOWED_DOCUMENT_TYPES

def validate_file_content(file_path: str, ext: str) -> None:
    """
    Validates actual file content, magic bytes, structure, and integrity.
    Prevents renamed binaries, corrupted files, encrypted documents, and resource exhaustion.
    Raises specific FileValidationError subclasses on failure.
    """
    import os
    import zipfile
    import fitz

    if not os.path.isfile(file_path):
        raise FileValidationError("Uploaded file does not exist on disk.")

    size = os.path.getsize(file_path)
    if size == 0:
        raise EmptyFileError("Uploaded file is empty (0 bytes).")

    if size > MAX_FILE_SIZE_BYTES:
        raise FileLimitExceededError(
            f"File size ({size / (1024 * 1024):.1f}MB) exceeds maximum limit of 50MB."
        )

    clean_ext = ext.lower().lstrip(".")

    if clean_ext == "pdf":
        with open(file_path, "rb") as f:
            header = f.read(1024)
        if b"%PDF-" not in header:
            raise CorruptedFileError(
                "File signature mismatch: expected PDF header (%PDF-). The file may be renamed or corrupted."
            )

        try:
            doc = fitz.open(file_path)
            if doc.is_encrypted or doc.needs_pass:
                doc.close()
                raise EncryptedFileError(
                    "Document is password-protected or encrypted. Please provide an unencrypted document."
                )
            page_count = len(doc)
            if page_count == 0:
                doc.close()
                raise EmptyFileError("PDF document contains zero pages.")
            if page_count > MAX_PAGE_COUNT:
                doc.close()
                raise FileLimitExceededError(
                    f"Document contains {page_count} pages, exceeding the limit of {MAX_PAGE_COUNT} pages."
                )
            doc.close()
        except (EncryptedFileError, EmptyFileError, FileLimitExceededError):
            raise
        except Exception as e:
            raise CorruptedFileError(f"PDF document is malformed or unreadable: {str(e)[:80]}")

    elif clean_ext == "docx":
        with open(file_path, "rb") as f:
            header = f.read(4)
        if header != b"PK\x03\x04":
            raise CorruptedFileError(
                "File signature mismatch: expected DOCX/ZIP header (PK..). The file may be renamed or corrupted."
            )

        if not zipfile.is_zipfile(file_path):
            raise CorruptedFileError("DOCX archive is corrupted or not a valid zip container.")

        try:
            with zipfile.ZipFile(file_path, "r") as z:
                names = set(z.namelist())
                if "word/document.xml" not in names:
                    raise CorruptedFileError("Invalid DOCX structure: missing word/document.xml.")
        except CorruptedFileError:
            raise
        except Exception as e:
            raise CorruptedFileError(f"DOCX archive is damaged or unreadable: {str(e)[:80]}")

    elif clean_ext == "txt":
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            raise CorruptedFileError(
                "Binary data detected in text file: contains null bytes and is not valid plain text."
            )
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError:
            try:
                chunk.decode("latin-1")
            except Exception:
                raise CorruptedFileError("Text file has unsupported or corrupt encoding.")

    elif clean_ext in ("png", "jpg", "jpeg"):
        with open(file_path, "rb") as f:
            header = f.read(16)
        if clean_ext == "png":
            if not header.startswith(b"\x89PNG\r\n\x1a\n"):
                raise CorruptedFileError(
                    "File signature mismatch: expected PNG header. The file may be renamed or corrupted."
                )
        elif clean_ext in ("jpg", "jpeg"):
            if not header.startswith(b"\xff\xd8\xff"):
                raise CorruptedFileError(
                    "File signature mismatch: expected JPEG header. The file may be renamed or corrupted."
                )

        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                w, h = img.size
                if w == 0 or h == 0:
                    raise EmptyFileError("Image document has 0 width or height.")
                if w > 10000 or h > 10000:
                    raise FileLimitExceededError(
                        f"Image dimensions ({w}x{h}) exceed maximum allowable size of 10000x10000 pixels."
                    )
        except (EmptyFileError, FileLimitExceededError):
            raise
        except Exception as img_err:
            raise CorruptedFileError(f"Image document is malformed or unreadable: {str(img_err)[:80]}")

def sanitize_chat_input(user_input: str) -> tuple[str, bool, list[str]]:
    """
    Best-effort defensive scan for prompt-injection patterns in user input.
    Returns: (sanitized_text, was_flagged, detected_patterns)
    """
    detected = []
    sanitized = user_input
    
    for pattern in PROMPT_INJECTION_PATTERNS:
        matches = re.findall(pattern, sanitized)
        if matches:
            detected.append(pattern)
            logger.warning(f"Prompt injection pattern detected: {pattern} in user input")
            sanitized = re.sub(pattern, "[DEFANGED_INSTRUCTION]", sanitized)
            
    was_flagged = len(detected) > 0
    return sanitized, was_flagged, detected
