from typing import List, Dict, NoReturn
import os
import fitz
import requests
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
IMAGE_CAPTIONING_MODEL_URL = os.getenv("IMAGE_CAPTIONING_MODEL_URL")


def extract_images_from_pdf(pdf_path: str, output_folder: str) -> List[Dict[str, str]]:
    """
    Extract images from a PDF file and return a list of dictionaries with the image file path and the page number.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_folder (str): The directory where the extracted images will be saved.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary has the keys "image_location" and "page_of_image".
    """
    image_info = []

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

            # Add the image information to the list
            image_info.append({
                "image_location": image_filename,
                "page_of_image": page_num + 1
            })
            print(f"Extracted image {image_index} from page {page_num + 1} to {image_filename}")

    # Close the PDF file
    doc.close()

    return image_info

def caption_image(filename:str)-> List[Dict]:
    """
    Captions an image using the Hugging Face Hub API.

    Args:
        filename (str): The path to the image file to be captioned.

    Returns:
        List[Dict]: A list of dictionaries containing caption information for the image.
    """
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    
    with open(filename, "rb") as f:
        # Reset the file pointer to the beginning of the file
        # f.seek(0)
        data = f.read()
        
    response = requests.post(IMAGE_CAPTIONING_MODEL_URL, headers=headers, data=data)
    return response.json()