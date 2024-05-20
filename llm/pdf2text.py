from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path based on your Tesseract installation

# Path to the PDF file
pdf_path = 'page2.pdf'

# Convert PDF to images
pages = convert_from_path(pdf_path, 300)

# Directory to save the images
image_dir = './images'
os.makedirs(image_dir, exist_ok=True)

# Process each page
for page_number, page in enumerate(pages):
    image_path = os.path.join(image_dir, f'page_{page_number + 1}.png')
    page.save(image_path, 'PNG')

    # Extract text from the image using Tesseract
    text = pytesseract.image_to_string(Image.open(image_path), lang='tel')
    
    # Print or save the extracted text
    print(f"Text from page {page_number + 1}:")
    print(text)
    print("\n" + "="*80 + "\n")

    # Optionally, save the text to a file
    with open(f'extracted_text_page_{page_number + 1}.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(text)

