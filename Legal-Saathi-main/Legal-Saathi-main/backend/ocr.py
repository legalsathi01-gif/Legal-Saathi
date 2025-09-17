# backend/ocr.py
from PIL import Image
import pytesseract
import cv2  # Import OpenCV
import numpy as np  # Import NumPy


def ocr_image_with_preprocessing(pil_image):
    """
    Performs OCR on a PIL image after applying preprocessing with OpenCV.
    """
    # 1. Convert the PIL Image to an OpenCV format (NumPy array)
    open_cv_image = np.array(pil_image)

    # 2. Convert the image to grayscale
    # 1. Ensure the image is in a format OpenCV can handle (RGB)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
        
    # 2. Convert the PIL Image to an OpenCV format (NumPy array)
    # PIL is RGB, OpenCV is BGR, so we convert color channels
    open_cv_image = np.array(pil_image)[:, :, ::-1]
 
    # 3. Convert the image to grayscale
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # 3. Apply adaptive thresholding to create a clean black-and-white image
    # 4. Apply adaptive thresholding to create a clean black-and-white image
    # This helps Tesseract distinguish text from the background
    preprocessed_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 4. Pass the clean, preprocessed image to Tesseract
    # 5. Pass the clean, preprocessed image to Tesseract
    # We add a configuration option (--psm 6) to assume a single uniform block of text
    try:
        text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--psm 6')
    except pytesseract.TesseractNotFoundError:
        error_msg = (
            "Tesseract Error: The Tesseract executable was not found. "
            "Please install Tesseract from the official website and ensure it is in your system's PATH. "
            "The application will not be able to read text from images until this is resolved."
        )
        print(error_msg)
        # Return the error message to be displayed in the UI
        return error_msg

    # Basic clean-up
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text


# --- Test Block ---
if __name__ == "__main__":
    try:
        img = Image.open("photo.jpeg") # Make sure a test image named 'photo.jpeg' exists
        extracted_text = ocr_image_with_preprocessing(img)
        print("--- Extracted Text ---")
        print(extracted_text)
    except FileNotFoundError:
        print("Error: 'photo.jpeg' not found. Please place a test image in the 'backend' folder.")
