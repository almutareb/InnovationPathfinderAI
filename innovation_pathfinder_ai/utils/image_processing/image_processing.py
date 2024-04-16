import os
from typing import NoReturn
import fitz

def extract_images_from_pdf(pdf_path: str, output_folder: str) -> NoReturn:
    """
    Extract images from a PDF file and save them as individual image files.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_folder (str): The directory where the extracted images will be saved.

    Returns:
        None
    """
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Loop through each page in the PDF
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images()

        # Loop through the images on the page and save them to files
        for image_index, img in enumerate(images, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes: bytes = base_image["image"]
            image_ext: str = base_image["ext"]

            # Save the image to a file
            os.makedirs(output_folder, exist_ok=True)
            image_filename: str = os.path.join(output_folder, f"page{page_num + 1}_image{image_index}.{image_ext}")
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)

            print(f"Extracted image {image_index} from page {page_num + 1} to {image_filename}")

    # Close the PDF file
    doc.close()