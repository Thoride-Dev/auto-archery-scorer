import unittest
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from archery_scorer.preprocessor import ImagePreprocessor

class TestImagePreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method will be executed once before any test is run
        # Load a test image or create a synthetic one for testing purposes
        cls.test_image_path = 'tests/data/test_image_02.jpeg'  # Provide a valid path to a test image
        cls.test_image = cv2.imread(cls.test_image_path, cv2.IMREAD_COLOR)
        if cls.test_image is None:
            raise FileNotFoundError(f"Test image not found at {cls.test_image_path}")
        cls

    def show_image(self, image, title="Image", wait_key_time=0, scale=0.75):
        """
        Display the image in a window with optional scaling.
        :param image: The image to be displayed.
        :param title: The title of the window.
        :param wait_key_time: Time in milliseconds to wait for a key event. 0 means wait indefinitely.
        :param scale: Scaling factor for the image size.
        """
        if scale != 1.0:
            # Calculate the new dimensions based on the scaling factor
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            # Resize the image using the new dimensions
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = image

        cv2.imshow(title, scaled_image)
        cv2.waitKey(wait_key_time)
        cv2.destroyAllWindows()
    
    def test_load_image(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        loaded_image = preprocessor.load_image()
        self.assertIsNotNone(loaded_image, "Loaded image should not be None")
        # self.show_image(loaded_image)
    
    def test_apply_gaussian_blur(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        blurred_image = preprocessor.apply_gaussian_blur()
        self.assertIsNotNone(blurred_image, "Blurred image should not be None")
        # self.show_image(blurred_image)
    
    def test_detect_edges(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        edges = preprocessor.detect_edges(self.test_image)
        self.assertIsNotNone(edges, "Edges should be detected")
        # self.show_image(edges)
    
    def test_correct_perspective(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        # Provide a set of corners for perspective correction
        # These corners should be determined based on the actual test image
        corners = np.float32([[150, 300], [720, 195], [760, 980], [20, 900]])
        rectified_image = preprocessor.correct_perspective(corners)
        self.assertIsNotNone(rectified_image, "Rectified image should not be None")
        # self.show_image(rectified_image)
    
    def test_detect_corners(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        corners = preprocessor.detect_corners()
        self.assertEqual(len(corners), 4, "Should detect exactly 4 corners")

    def test_preprocessing_pipeline(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        # Load the image and convert it to grayscale
        grayscale_image = preprocessor.load_image()
        self.assertIsNotNone(grayscale_image, "Grayscale image should not be None")
        # Optionally display the grayscale image
        self.show_image(grayscale_image, title="Grayscale Image", wait_key_time=0)

        # Apply Gaussian blur
        blurred_image = preprocessor.apply_gaussian_blur()
        self.assertIsNotNone(blurred_image, "Blurred image should not be None")
        # Optionally display the blurred image
        self.show_image(blurred_image, title="Blurred Image", wait_key_time=0)

        # Detect edges
        edges = preprocessor.detect_edges(blurred_image)
        self.assertIsNotNone(edges, "Edges should be detected")
        # Optionally display the edge-detected image
        self.show_image(edges, title="Edges", wait_key_time=0)

        # Detect corners
        corners = preprocessor.detect_corners()
        self.assertEqual(len(corners), 4, "Should detect exactly 4 corners")
        print(corners)

        # Draw the detected corners on the original image
        original_image_with_corners = preprocessor.original_image.copy()
        for corner in corners:
            cv2.drawMarker(original_image_with_corners, tuple(int(v) for v in corner), (0, 255, 0), cv2.MARKER_CROSS, markerSize=20, thickness=2)
        self.show_image(original_image_with_corners, title="Original Image With Corners", wait_key_time=0)

        # Correct perspective
        rectified_image = preprocessor.correct_perspective(corners)
        self.assertIsNotNone(rectified_image, "Rectified image should not be None")
        # Optionally display the rectified image
        self.show_image(rectified_image, title="Rectified Image", wait_key_time=0)

if __name__ == '__main__':
    unittest.main()
