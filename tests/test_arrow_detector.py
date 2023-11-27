import unittest
import cv2
import numpy as np
import sys
import os

# Adjust the path to import the necessary modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from archery_scorer.target_detector import TargetDetector
from archery_scorer.preprocessor import ImagePreprocessor
from archery_scorer.arrow_detector import ArrowDetector

class TestArrowDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method will be executed once before any test is run
        # Load a test image or create a synthetic one for testing purposes
        cls.test_image_path = 'tests/data/7-01.jpg'  # Provide a valid path to a test image
        
        # Preprocess the image to obtain rectified image and circles
        cls.preprocessor = ImagePreprocessor(image_path = cls.test_image_path)
        cls.preprocessed_image = cls.preprocessor.detect_and_correct_ovals()
        for i in range(5):
            cls.preprocessor = ImagePreprocessor(image = cls.preprocessed_image)
            cls.preprocessed_image = cls.preprocessor.detect_and_correct_ovals()

        target_detector = TargetDetector(cls.preprocessed_image)
        # Detect circles on the target face
        circles = target_detector.detect_circles()
        
        # Create an instance of ArrowDetector with the rectified image and circles
        cls.arrow_detector = ArrowDetector(cls.preprocessed_image, circles)

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

    def test_detect_arrow(self):
        # Detect arrow lines on the test image
        arrow_line, lines = self.arrow_detector.detect_arrow()
        
        # Assert that arrow lines have been detected
        self.assertIsNotNone(arrow_line, "Arrow line should be detected")
        
        # Optionally, visualize the detected arrow line on the image
        if arrow_line is not None:
            x1, y1, x2, y2 = arrow_line
            cv2.line(self.preprocessed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            self.show_image(self.preprocessed_image, title="Detected Arrow Line", wait_key_time=0)

        print(arrow_line)

        # Draw all the lines detected
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(self.preprocessed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        self.show_image(self.preprocessed_image, title="Detected Lines", wait_key_time=0)

if __name__ == '__main__':
    unittest.main()
