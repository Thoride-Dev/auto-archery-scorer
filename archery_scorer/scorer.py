import cv2
import numpy as np

class Scorer:
    """
    This class is responsible for scoring each arrow based on its position on the target.
    """
    
    def __init__(self, arrows, circles):
        """
        Initialize the Scorer with detected arrows and circles.
        """
        self.arrows = arrows
        self.circles = circles
    
    def score_arrow(self, arrow):
        """
        Score a single arrow based on its position relative to the circles.
        """
        pass
    
    def calculate_total_score(self):
        """
        Calculate the total score for all detected arrows.
        """
        pass