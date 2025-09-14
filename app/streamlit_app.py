# backend/ocr.py
from PIL import Image
import pytesseract

def ocr_image_pil(image: Image.Image) -> str:
    """
    Performs Optical Character Recognition (OCR) on a PIL Image object.

    Args:
        image (Image.Image): The image to process.

    Returns:
        str: The extracted text. Returns an error message if Tesseract is not found.
    """
    try:
        # If Tesseract is not in your system's PATH (especially on Windows),
        # you might need to specify its location.
        # Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(image)
        return text
    except pytesseract.TesseractNotFoundError:
        error_msg = "Error: Tesseract is not installed or not in your PATH. OCR cannot be performed."
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during OCR: {e}"
        print(error_msg)
        return error_msg
