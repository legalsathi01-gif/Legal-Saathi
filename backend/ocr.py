# backend/ocr.py
from PIL import Image
import pytesseract
import cv2  # Import OpenCV
import numpy as np  # Import NumPy

# Make sure this path is correct
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_image_with_preprocessing(pil_image):
    """
    Performs OCR on a PIL image after applying preprocessing with OpenCV.
    """
    # 1. Convert the PIL Image to an OpenCV format (NumPy array)
    open_cv_image = np.array(pil_image)

    # 2. Convert the image to grayscale
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # 3. Apply adaptive thresholding to create a clean black-and-white image
    # This helps Tesseract distinguish text from the background
    preprocessed_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 4. Pass the clean, preprocessed image to Tesseract
    # We add a configuration option (--psm 6) to assume a single uniform block of text
    text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--psm 6')

    # Basic clean-up
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text


# --- Your code to run the function ---
try:
    img = Image.open("photo.jpeg")
    extracted_text = ocr_image_with_preprocessing(img)

    if extracted_text:
        print("--- Extracted Text ---")
        print(extracted_text)
    else:
        print("--- No text found in the image after preprocessing. ---")

except FileNotFoundError:
    print("Error: photo.jpeg not found. Make sure it's in the 'Backend' folder.")
