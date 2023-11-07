import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class ImagePreprocessor:
    """
    This class is responsible for preprocessing the input image to prepare it for further analysis.
    This includes noise reduction, edge detection, and perspective correction.
    """
    
    def __init__(self, image_path):
        """
        Initialize the ImagePreprocessor with the path to the image.
        """
        self.image_path = image_path
        self.original_image = self.load_image()

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
    
    def load_image(self):
        """
        Load the image from the given path, convert it to grayscale, and scale it down to fit within 1024x1024.
        """
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image at {self.image_path} cannot be loaded.")
        
        # Calculate the scaling factor to fit the image within 1024x1024
        max_dimension = max(image.shape[:2])
        scale_factor = 1024 / max_dimension if max_dimension > 1024 else 1
        
        # Resize the image if necessary
        if scale_factor != 1:
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    
    def apply_gaussian_blur(self, kernel_size=(3, 3), sigma=3):
        """
        Apply Gaussian blur to the image to reduce noise.
        :param kernel_size: Size of the Gaussian kernel.
        :param sigma_x: Gaussian kernel standard deviation in X direction.
        """
        blurred_image = cv2.GaussianBlur(self.original_image, kernel_size, sigmaX=sigma, sigmaY=sigma)
        return blurred_image
    
    def detect_edges(self, image, low_threshold=30, high_threshold=200):
        """
        Use Canny edge detection to find edges in the image.
        :param low_threshold: Lower bound for the hysteresis thresholding.
        :param high_threshold: Upper bound for the hysteresis thresholding.
        """
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges
    
    def correct_perspective(self, corners, output_size=(1024, 1024)):
        """
        Correct the perspective of the image using the detected corners.
        :param corners: Coordinates of the corners of the target in the image.
        :param output_size: Desired size of the output rectified image.
        """
        # Define the points to which the corners will be mapped.
        # These points form a square with the desired output size.
        dst_points = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1]
        ], dtype=np.float32)
        
        # Compute the perspective transform matrix and apply it to the image.
        transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
        rectified_image = cv2.warpPerspective(self.original_image, transform_matrix, output_size)
        return rectified_image
    
    def detect_corners(self, blur_kernel_size=(5, 5), sobel_kernel_size=3, harris_block_size=7, harris_ksize=7, harris_k=0.06, threshold=0.2):
        """
        Detect the corners of the target face using the Harris corner detector.
        :param blur_kernel_size: Kernel size for the Gaussian blur preprocessing.
        :param sobel_kernel_size: Aperture parameter for the Sobel operator.
        :param harris_block_size: Neighborhood size for Harris corner detection.
        :param harris_ksize: Aperture parameter for the Harris corner detection.
        :param harris_k: Harris detector free parameter.
        :param threshold: Threshold for detecting strong corners.
        """
        # Preprocess the image with Gaussian blur
        blurred_image = self.apply_gaussian_blur(kernel_size=blur_kernel_size)

        # Apply the Harris corner detector
        harris_response = cv2.cornerHarris(blurred_image, harris_block_size, harris_ksize, harris_k)
        
        # Dilate the Harris response to merge corner regions
        harris_response = cv2.dilate(harris_response, None)

        # Threshold the normalized response to get the corners
        corners = np.where(harris_response > threshold * harris_response.max())
        
        # Convert the coordinates to (x, y) pairs
        corners = np.float32(list(zip(corners[1], corners[0])))

        # Refine the corner locations to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
        #corners = cv2.cornerSubPix(blurred_image, corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)

        original_image_with_corners = self.original_image.copy()
        for corner in corners:
            cv2.drawMarker(original_image_with_corners, tuple(int(v) for v in corner), (0, 255, 0), cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
        self.show_image(original_image_with_corners, title="Original Image With Corners", wait_key_time=0)

        # Filter corners to find the four most prominent ones
        filtered_corners = self.filter_corners(corners)
        
        return filtered_corners

    def filter_corners(self, corners, max_corners=4):
        """
        Filter the detected corners to find the four corners of the target paper.
        :param corners: Detected corners from the Harris corner detector.
        :param max_corners: The maximum number of corners to return.
        """
         # Apply DBSCAN clustering to group corners that are close to each other
        dbscan = DBSCAN(eps=50, min_samples=20)
        labels = dbscan.fit_predict(corners)

        # Calculate the centroid of each cluster
        clustered_corners = []
        for label in set(labels):
            if label != -1:  # Ignore noise points labeled as -1
                cluster_points = corners[labels == label]
                centroid = np.mean(cluster_points, axis=0)
                clustered_corners.append(centroid)

        # Convert the list of centroids to a NumPy array
        clustered_corners = np.array(clustered_corners, dtype=np.float32)
        corners = clustered_corners
        # If there are more than max_corners, we need to select the best ones
        if len(corners) > max_corners:
            # One way to select the best corners is to use a heuristic such as distance from the center
            centroid = np.mean(corners, axis=0)
            distances = np.sqrt(np.sum((corners - centroid)**2, axis=1))
            # Sort corners by their distance to the centroid (farthest first)
            corners = corners[np.argsort(-distances)]
        
        # Keep only the max_corners number of points
        corners = corners[:max_corners]
        
        # Calculate the centroid of the corners
        centroid = np.mean(corners, axis=0)
        
        # Sort the corners based on their angle with respect to the centroid
        def angle_with_centroid(corner):
            return np.arctan2(corner[1] - centroid[1], corner[0] - centroid[0])
        
        corners = sorted(corners, key=angle_with_centroid)

        # Check if the sorted corners form a convex quadrilateral
        if not self.is_convex_quadrilateral(corners):
            raise ValueError("The detected corners do not form a convex quadrilateral.")
        
        # Return the filtered corners
        return np.array(corners, dtype=np.float32)
    
    def is_convex_quadrilateral(self, corners):
        """
        Check if four points form a convex quadrilateral.
        :param corners: Four corners sorted in a consistent order.
        """
        # Calculate cross product of adjacent edges around the quadrilateral
        def cross_product(a, b, c):
            ab = b - a
            bc = c - b
            return np.cross(ab, bc)
        
        cp1 = cross_product(corners[0], corners[1], corners[2])
        cp2 = cross_product(corners[1], corners[2], corners[3])
        cp3 = cross_product(corners[2], corners[3], corners[0])
        cp4 = cross_product(corners[3], corners[0], corners[1])
        
        # Check if cross products have the same sign (all positive or all negative)
        return np.sign(cp1) == np.sign(cp2) == np.sign(cp3) == np.sign(cp4)