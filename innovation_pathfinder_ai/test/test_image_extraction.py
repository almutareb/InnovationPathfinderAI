import os
import shutil
import unittest
from unittest.mock import patch
from typing import List, Dict
import tempfile
import requests

from innovation_pathfinder_ai.utils.image_processing.image_processing import extract_images_from_pdf

class TestExtractImagesFromPDF(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.pdf_file_path = os.path.join(self.tmp_dir.name, "attention_is_all_you_need.pdf")
        self.download_pdf_file()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def download_pdf_file(self):
        url = "https://arxiv.org/pdf/1706.03762.pdf"
        response = requests.get(url)
        with open(self.pdf_file_path, "wb") as file:
            file.write(response.content)

    def test_extract_images_from_pdf_default_output_dir(self):
        # Test the function with the default output directory
        output_folder = os.path.join(self.tmp_dir.name, 'images')

        image_info = extract_images_from_pdf(self.pdf_file_path, output_folder)
        self.assertGreater(len(os.listdir(output_folder)), 0)
        self.assertIsInstance(image_info, list)
        for file_info in image_info:
            self.assertIsInstance(file_info, dict)
            self.assertIn("image_location", file_info)
            self.assertIn("page_of_image", file_info)
            self.assertTrue(os.path.exists(file_info["image_location"]))

        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
        os.rmdir(output_folder)


    def test_extract_images_from_pdf_non_existent_output_dir(self):
        # Test the function with a non-existent output directory
        output_folder = os.path.join(self.tmp_dir.name, 'non_existent_dir', 'images')

        image_info = extract_images_from_pdf(self.pdf_file_path, output_folder)
        self.assertGreater(len(os.listdir(os.path.join(self.tmp_dir.name, 'non_existent_dir'))), 0)
        for file_info in image_info:
            self.assertIsInstance(file_info, dict)
            self.assertIn("image_location", file_info)
            self.assertIn("page_of_image", file_info)
            self.assertTrue(os.path.exists(file_info["image_location"]))

        shutil.rmtree(os.path.join(self.tmp_dir.name, 'non_existent_dir'))
        
        
    # def test_extract_images_from_pdf_custom_output_dir(self):
    #         # Test the function with a custom output directory
    #         output_folder = os.path.join(self.tmp_dir.name, 'custom_images')

    #         image_info = extract_images_from_pdf(self.pdf_file_path, output_folder)
    #         self.assertGreater(len(os.listdir(output_folder)), 0)
    #         self.assertIsInstance(image_info, List[Dict[str, str]])
    #         for file_info in image_info:
    #             self.assertIn("image_location", file_info)
    #             self.assertIn("page_of_image", file_info)
    #             self.assertTrue(os.path.exists(file_info["image_location"]))

    #         for file in os.listdir(output_folder):
    #             os.remove(os.path.join(output_folder, file))
    #         os.rmdir(output_folder)

    # @patch('fitz.Page.get_images', return_value=[])
    # def test_extract_images_from_pdf_no_images(self, mock_get_images):
    #     # Test the function with a PDF file that has no images
    #     output_folder = os.path.join(self.tmp_dir.name, 'no_images')

    #     image_info = extract_images_from_pdf(self.pdf_file_path, output_folder)
    #     self.assertEqual(len(os.listdir(output_folder)), 0)
    #     self.assertIsInstance(image_info, List[Dict[str, str]])
    #     self.assertEqual(len(image_info), 0)

    #     shutil.rmtree(output_folder)