import cv2
import numpy as np

class ArrowDetector:
    """
    This class is responsible for detecting arrows within the target face.
    """
    
    def __init__(self, rectified_image, circles):
        """
        Initialize the ArrowDetector with the rectified image and detected circles.
        """
        self.rectified_image = rectified_image
        self.circles = circles
    
    def detect_gaps(self):
        """
        Detect gaps in the concentric circles that may indicate the presence of arrows.
        """
        gaps = []
        if len(self.circles) > 1:
            for i in range(len(self.circles) - 1):
                # Calculate the distance between circles
                distance = np.abs(self.circles[i][0][2] - self.circles[i + 1][0][2])
                gaps.append(distance)
        return gaps
    
    def detect_arrow(self):
        """
        Detect a single arrow by finding lines within the target face.
        """
        # Edge detection using Canny
        edges = cv2.Canny(self.rectified_image, 50, 150)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        
        # Extracting line coordinates excluding those with center points near the image boundaries
        line_coords = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate center point of the line
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # Filter out lines with center points too close to the image boundaries
                if center_x > 100 and center_x < self.rectified_image.shape[1] - 100 and \
                center_y > 100 and center_y < self.rectified_image.shape[0] - 100:
                    line_coords.append([x1, y1, x2, y2])
        
        # Fit a line to the arrow among the filtered lines
        if len(line_coords) > 0:
            # Assuming the arrow will be the longest line detected
            longest_line = max(line_coords, key=lambda line: np.linalg.norm(np.array(line[:2]) - np.array(line[2:])))
            return longest_line, line_coords
        else:
            return None
