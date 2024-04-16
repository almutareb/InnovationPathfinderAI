import os
import shutil
import pytest
import requests

from innovation_pathfinder_ai.utils.image_processing.image_processing import (
    extract_images_from_pdf
)

import os
import shutil
import fitz
from unittest.mock import patch
import pytest
import requests
from pathlib import Path
import os
import shutil
import fitz
from unittest.mock import patch
import pytest
import requests
from pathlib import Path

@pytest.fixture
def pdf_file(tmp_path):
    pdf_file_path = tmp_path / "attention_is_all_you_need.pdf"
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    response = requests.get(url)
    with open(pdf_file_path, "wb") as file:
        file.write(response.content)
    return pdf_file_path

def test_extract_images_from_pdf_default_output_dir(pdf_file, tmp_path):
    # Test the function with the default output directory
    output_folder = os.path.join(tmp_path, 'images')
    extract_images_from_pdf(pdf_file, output_folder)
    assert len(os.listdir(output_folder)) > 0
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))
    os.rmdir(output_folder)

def test_extract_images_from_pdf_custom_output_dir(pdf_file, tmp_path):
    # Test the function with a custom output directory
    output_folder = os.path.join(tmp_path, 'custom_images')
    extract_images_from_pdf(pdf_file, output_folder)
    assert len(os.listdir(output_folder)) > 0
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))
    os.rmdir(output_folder)

def test_extract_images_from_pdf_non_existent_output_dir(pdf_file, tmp_path):
    # Test the function with a non-existent output directory
    output_folder = os.path.join(tmp_path, 'non_existent_dir', 'images')
    extract_images_from_pdf(pdf_file, output_folder)
    assert len(os.listdir(os.path.join(tmp_path, 'non_existent_dir'))) > 0
    shutil.rmtree(os.path.join(tmp_path, 'non_existent_dir'))

# def test_extract_images_from_pdf_no_images(pdf_file, tmp_path):
#     # Test the function with a PDF file that has no images
#     with patch('fitz.Page.get_images', return_value=[]):
#         output_folder = os.path.join(tmp_path, 'no_images')
#         extract_images_from_pdf(pdf_file, output_folder)
#         assert len(os.listdir(output_folder)) == 0
#         shutil.rmtree(output_folder)