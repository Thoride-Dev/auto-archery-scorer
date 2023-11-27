import cv2
import numpy as np


class TargetDetector:
    """
    This class is responsible for detecting the target face and its concentric circles.
    """
    
    def __init__(self, preprocessed_image):
        """
        Initialize the TargetDetector with the preprocessed image.
        """
        self.preprocessed_image = preprocessed_image
    
    def find_target_corners(self):
        """
        Detect the corners of the target face.
        """
        # Assuming the target face is the largest square-like contour
        # This method would be implemented if needed
        pass
    
    def rectify_target(self):
        """
        Rectify the image so that the target face is frontal and circular.
        """
        # Assuming the preprocessor already rectifies the image to a circle
        # This method would be implemented if needed
        pass
    
    def detect_circles(self, dp=1.2, minDist=0.001, param1=150, param2=700, minRadius=1, maxRadius=2000):
        """
        Detect the concentric circles of the target face using Hough Circle Transform.
        :param dp: Inverse ratio of the accumulator resolution to the image resolution.
        :param minDist: Minimum distance between the centers of the detected circles.
        :param param1: Higher threshold for the Canny edge detector.
        :param param2: Threshold for center detection.
        :param minRadius: Minimum circle radius.
        :param maxRadius: Maximum circle radius.
        :return: A list of detected circles with (x, y, radius).
        """
        #gray_image = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2GRAY)
        image = self.preprocessed_image.copy()

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (3, 3), 2)

        cv2.imshow("As", blurred_image)
        cv2.imwrite("blurred.jpg", blurred_image)
        
        # Apply edge detection
        edges = cv2.Canny(blurred_image, 300, 600)

        # Apply morphological operations to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=3)
        cv2.imshow("as", edges)
        cv2.imwrite("edges.jpg", edges)

        circles = None
        while circles is None or len(circles[0])<10:
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius
            )
            param2-=5
        
        print(circles)
        if circles is not None:
            print(f"Number of Circles: {len(circles[0])}")
            circles = np.round(circles[0, :]).astype("int")
            return circles
        else:
            return []

# Example usage:
# preprocessed_image = ... # This should be the output from the preprocessor
# target_detector = TargetDetector(preprocessed_image)
# circles = target_detector.detect_circles()