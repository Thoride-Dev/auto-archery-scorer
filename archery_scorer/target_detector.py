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

    def group_similar_circles(self, circles, radius_threshold):
        """
        Group circles with similar radii.
        """
        circles = sorted(circles, key=lambda x: x[2], reverse=True)
        grouped_circles = []
        skip_indices = set()

        for i, circle in enumerate(circles):
            if i in skip_indices:
                continue
            similar_circles = [circle]
            for j, other_circle in enumerate(circles[i+1:], start=i+1):
                if abs(circle[2] - other_circle[2]) <= radius_threshold:
                    similar_circles.append(other_circle)
                    skip_indices.add(j)
            # Combine similar circles by averaging their coordinates and radii
            average_circle = np.mean(similar_circles, axis=0).astype(int)
            grouped_circles.append(average_circle)

        return grouped_circles

    
    def detect_circles(self, dp=0.75, minDist=0.000000001, param1=150, param2=700, minRadius=1, maxRadius=2000, radius_threshold=50):
        """
        Detect the concentric circles of the target face using Hough Circle Transform.
        """
        image = self.preprocessed_image.copy()
        blurred_image = cv2.GaussianBlur(image, (5, 5), 2)
        all_circles = []

        circles = None
        while circles is None or len(circles[0])<15:
            circles = cv2.HoughCircles(
                blurred_image,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius
            )
            param2-=5

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Combine all circles onto the original image
            image_c = self.preprocessed_image.copy()
            for circle in circles:
                cv2.circle(image_c, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
            cv2.imshow(f"asd",image_c)
            # Group circles with similar radii
            circles = self.group_similar_circles(circles, radius_threshold)

        return circles

# Example usage:
# preprocessed_image = ... # This should be the output from the preprocessor
# target_detector = TargetDetector(preprocessed_image)
# circles = target_detector.detect_circles()