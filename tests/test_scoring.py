import unittest
import cv2
import sys
import os

# Import the ArrowDetector and ArrowEvaluation classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from archery_scorer.arrow_detector import ArrowDetector
from archery_scorer.preprocessor import ImagePreprocessor
from archery_scorer.scorer import ArrowScorer
from archery_scorer.target_detector import TargetDetector

class TestArrowScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method will be executed once before any test is run
        # Load a test image or create a synthetic one for testing purposes
        cls.test_image_path = 'tests/data/6-01.jpg'  # Provide a valid path to a test image

        # Preprocess the image to obtain rectified image and circles
        cls.preprocessor = ImagePreprocessor(image_path = cls.test_image_path)
        cls.preprocessed_image = cls.preprocessor.detect_and_correct_ovals()
        for i in range(0):
            cls.preprocessor = ImagePreprocessor(image = cls.preprocessed_image)
            cls.preprocessed_image = cls.preprocessor.detect_and_correct_ovals()
        
        # Perform circle detection 
        cls.target_detector = TargetDetector(cls.preprocessed_image)
        # Detect circles on the target face
        cls.circles = cls.target_detector.detect_circles()

        # Perform arrow detection
        cls.arrow_detector = ArrowDetector(cls.preprocessed_image, cls.circles)
        cls.arrow_line, cls.lines = cls.arrow_detector.detect_arrow()

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

    def test_evaluate_arrow_score(self):
        # Create an instance of ArrowScorer with the circles and arrow line
        scorer = ArrowScorer(self.circles, self.arrow_line)
        # Evaluate the arrow score
        arrow_score = scorer.evaluate_arrow()
        print("Arrow score:", arrow_score)

        # Display the image with the arrow line and circles
        image_with_arrow_line = self.preprocessed_image.copy()
        cv2.line(image_with_arrow_line, (self.arrow_line[0], self.arrow_line[1]), (self.arrow_line[2], self.arrow_line[3]), (255, 0, 0), 2)
        for circle in self.circles:
            cv2.circle(image_with_arrow_line, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
        self.show_image(image_with_arrow_line, title="Detected Arrow Line", wait_key_time=0)

        # Assert the expected arrow score based off test image name
        expected_arrow_score = os.path.basename(self.test_image_path).split('-')[0]
        self.assertEqual(str(arrow_score), expected_arrow_score, "Arrow score should match the expected value")
        
if __name__ == '__main__':
    unittest.main()
