import unittest
import cv2
import sys
import os
import glob

# Import the necessary classes as before
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from archery_scorer.arrow_detector import ArrowDetector
from archery_scorer.preprocessor import ImagePreprocessor
from archery_scorer.scorer import ArrowScorer
from archery_scorer.target_detector import TargetDetector

class TestArrowScoring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_folder = 'data/Test_Cropped'
        cls.score_folders = ['6', '7', '8', '9', '10', 'M']
        cls.image_paths = []
        for score_folder in cls.score_folders:
            folder_path = os.path.join(cls.data_folder, score_folder)
            cls.image_paths.extend(glob.glob(os.path.join(folder_path, '*.jpg')))
        cls.results = []
        cls.correct = 0
        cls.tested = 0

    def test_evaluate_all_arrow_scores(self):
        
        i = 0
        for image_path in self.image_paths:
            i+=1
            print(i)
            with self.subTest(image_path=image_path):
                preprocessor = ImagePreprocessor(image_path=image_path)
                preprocessed_image = preprocessor.detect_and_correct_ovals()

                target_detector = TargetDetector(preprocessor.original_image)
                circles = target_detector.detect_circles()

                arrow_detector = ArrowDetector(preprocessor.original_image, circles)
                arrow_line, lines = arrow_detector.detect_arrow()

                scorer = ArrowScorer(circles, arrow_line)
                arrow_score = scorer.evaluate_arrow()

                expected_score = os.path.basename(os.path.dirname(image_path))
                expected_score = '0' if expected_score == 'M' else expected_score

                print(f"{int(arrow_score)} ?= {int(expected_score)}")
                self.tested+=1
                if arrow_score == expected_score:
                    self.correct+=1
                result = (image_path, arrow_score, expected_score)
                self.results.append(result)
                #self.assertEqual(str(arrow_score), expected_score, f"Arrow score for {image_path} should match the expected value")

        print(f"{self.correct}/{self.tested} = {self.correct/self.tested}")

    @classmethod
    def tearDownClass(cls):
        print(f"{cls.correct}/{cls.tested} = {cls.correct/cls.tested}")
        correct_count = sum(1 for result in cls.results if result[1] == result[2])
        total_count = len(cls.results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"\nAccuracy: {accuracy:.2f} ({correct_count}/{total_count})")

        print("\nDetailed Results:")
        for image_path, arrow_score, expected_score in cls.results:
            status = "Correct" if arrow_score == expected_score else "Incorrect"
            print(f"{image_path}: Scored {arrow_score}, Expected {expected_score} - {status}")

if __name__ == '__main__':
    unittest.main()