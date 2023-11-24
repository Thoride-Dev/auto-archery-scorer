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
        cls.test_image_path = 'tests/data/8-09.jpg'  # Provide a valid path to a test image
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

    def test_detect_ellipses(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        # Apply Canny edge detection to the image
        blurred_image = preprocessor.apply_gaussian_blur()
        edges = cv2.Canny(blurred_image, 150, 400)
        #self.show_image(edges, title="Canny Edges (Detect Ellipses)", wait_key_time=0)
        # Detect ellipses in the edge-detected image
        ellipses = preprocessor.detect_ellipses(edges)
        self.assertIsNotNone(ellipses, "Ellipses should be detected")
        self.assertGreater(len(ellipses), 0, "At least one ellipse should be detected")
        # Optionally display the image with detected ellipses
        image_with_ellipses = preprocessor.original_image.copy()
        for ellipse in ellipses:
            cv2.ellipse(image_with_ellipses, ellipse, (0, 255, 0), 2)
        #self.show_image(image_with_ellipses, title="Detected Ellipses", wait_key_time=0)

    def test_estimate_ellipse_to_circle_transformation(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        # Use a synthetic ellipse for testing
        ellipse = ((100, 100), (80, 50), 30)  # center, axes, angle
        transformation_matrix = preprocessor.estimate_ellipse_to_circle_transformation(ellipse)
        self.assertIsNotNone(transformation_matrix, "Transformation matrix should not be None")

    def test_detect_and_correct_ovals(self):
        preprocessor = ImagePreprocessor(self.test_image_path)
        # Perform edge detection
        blurred_image = preprocessor.apply_gaussian_blur()
        edges = cv2.Canny(blurred_image, 150, 600)
        self.show_image(edges, title="Canny Edges (Detect & Correct)", wait_key_time=0)
        # Detect ellipses
        ellipses = preprocessor.detect_ellipses(edges)
        image_with_ellipses = preprocessor.original_image.copy()
        for ellipse in ellipses:
            cv2.ellipse(image_with_ellipses, ellipse, (0, 255, 0), 2)
        self.show_image(image_with_ellipses, title="Detected Ellipses (Detect & Correct)", wait_key_time=0)
        self.assertGreater(len(ellipses), 0, "At least one ellipse should be detected")
        # Correct the perspective of the most prominent ellipse
        corrected_image = preprocessor.detect_and_correct_ovals(150, 400)
        self.assertIsNotNone(corrected_image, "Corrected image should not be None")
        # Optionally display the corrected image
        self.show_image(corrected_image, title="Corrected Image", wait_key_time=0)

if __name__ == '__main__':
    unittest.main()
