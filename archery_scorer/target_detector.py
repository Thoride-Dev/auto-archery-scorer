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

    
    def detect_circles(self, dp=0.70, minDist=0.000000001, param1=150, param2=700, minRadius=0, maxRadius=0, radius_threshold=70):
        """
        Detect the concentric circles of the target face using Hough Circle Transform.
        """
        image = self.preprocessed_image.copy()

        cv2.imwrite(f"results/Org Preprocessed.jpg", image)

        blurred_image = cv2.GaussianBlur(image, (5, 5), 2)

        circles = None
        while circles is None or len(circles[0]) < 50:
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
            param2 -= 5

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # Detect circles
            image_with_circles = self.preprocessed_image.copy()
            color_image = cv2.cvtColor(image_with_circles, cv2.COLOR_GRAY2BGR)
            for circle in circles:
                cv2.circle(color_image , (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
            cv2.imshow(f"Circles", color_image)
            cv2.imwrite(f"results/Detected Circles.jpg", color_image)

            # Group circles with similar radii
            circles = self.group_similar_circles(circles, radius_threshold)

            # group circles
            image_with_circles = self.preprocessed_image.copy()
            color_image = cv2.cvtColor(image_with_circles, cv2.COLOR_GRAY2BGR)
            for circle in circles:
                cv2.circle(color_image , (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
            cv2.imshow(f"Grouped Circles", color_image)
            cv2.imwrite(f"results/Grouped Circles.jpg", color_image)

            # Ensure circles is a NumPy array
            circles = np.array(circles)

            # Sort circles from largest to smallest by radius
            circles = circles[circles[:, 2].argsort()[::-1]]
            
            # Calculate the total number of circles and the radii of each circle
            total_circles = len(circles)
            radii = [circle[2] for circle in circles]

            # Calculate the average difference between the radii
            if total_circles > 1:
                radii_diffs = [abs(radii[i] - radii[i - 1]) for i in range(1, total_circles)]
                average_diff = sum(radii_diffs) / len(radii_diffs)
            else:
                average_diff = 0

            # Calculate the average difference between the radii and centers
            center_shifts = []
            for i in range(1, len(circles)):
                center_shifts.append((circles[i-1][:2] - circles[i][:2]).tolist())

            if center_shifts:
                average_center_shift = np.mean(center_shifts, axis=0)
            else:
                average_center_shift = np.array([0, 0])

            # Loop to add smaller circles until a total of 5 circles is reached
            while len(circles) < 5:
                smallest_circle = circles[-1]
                smaller_circle_radius = max(smallest_circle[2] - average_diff, 0)  # Ensure radius is not negative
                if smaller_circle_radius > 0:
                    # Calculate the new center by applying the average shift
                    new_center = smallest_circle[:2] - average_center_shift*(0.5)
                    smaller_circle = (int(new_center[0]), int(new_center[1]), int(smaller_circle_radius))
                    circles = np.vstack((circles, [smaller_circle]))
                else:
                    break  # Stop if the radius becomes zero or negative


        else:
            radii = []
            average_diff = 0

        # extrapolated circles
        image_with_circles = self.preprocessed_image.copy()
        color_image = cv2.cvtColor(image_with_circles, cv2.COLOR_GRAY2BGR)
        for circle in circles:
            cv2.circle(color_image , (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
        cv2.imshow(f"Extrapolated Circles", color_image)
        cv2.imwrite(f"results/Extrapolated Circles.jpg", color_image)

        return circles

# Example usage:
# preprocessed_image = ... # This should be the output from the preprocessor
# target_detector = TargetDetector(preprocessed_image)
# circles = target_detector.detect_circles()