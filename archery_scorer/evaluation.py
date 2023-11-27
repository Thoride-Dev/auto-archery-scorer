import numpy as np

class ArrowEvaluation:
    """
    Class for evaluating the score of an arrow based on the position of the arrow tip
    relative to the concentric circles.
    """
    def __init__(self, circles, arrow_line):
        """
        Initialize the ArrowEvaluation with the detected circles and arrow line.
        """
        self.circles = circles
        self.arrow_line = arrow_line

    def evaluate_arrow(self):
        """
        Evaluate the score of the arrow based on the position of the arrow tip
        """
        # Check if arrow line exists
        if self.arrow_line is None:
            return 'M'  # Outside all circles, denoted as 'M'

        # Extract coordinates of arrow line endpoints
        x1, y1, x2, y2 = self.arrow_line

        # Calculate the center point of the arrow tip (using the endpoint closer to the circles)
        if np.linalg.norm(np.array([x1, y1]) - np.array(self.circles[0][:2])) < \
                np.linalg.norm(np.array([x2, y2]) - np.array(self.circles[0][:2])):
            arrow_tip_x, arrow_tip_y = x1, y1
        else:
            arrow_tip_x, arrow_tip_y = x2, y2

        # Check which circle contains the arrow tip
        for i, circle in enumerate(self.circles):
            circle_x, circle_y, radius = circle
            distance_to_center = np.sqrt((circle_x - arrow_tip_x) ** 2 + (circle_y - arrow_tip_y) ** 2)
            if distance_to_center <= radius:
                return 10 - i  # Assign score based on ring (10 for outermost, 6 for innermost)

        return 'M'  # Outside all circles