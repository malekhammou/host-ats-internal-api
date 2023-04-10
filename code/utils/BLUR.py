import cv2
import numpy as np

def estimate_blur_laplacian(image_file):
    """
    Estimate the blur in an image file using Laplacian operator.

    This function takes an image file path as input and calculates the blur score of the image using the Laplacian operator.
    The function first reads the image file using OpenCV, converts it to grayscale, and applies the Laplacian operator to detect edges.
    The function then calculates the variance of the edge map to estimate the blur score of the image.

    Parameters:
    - image_file (str): The path of the image file to be processed.

    Return Value:
    - A float value representing the estimated blur score of the image.

    Dependencies:
    - The function requires the following libraries to be imported:
      - cv2: OpenCV library for reading image files and performing image processing operations.
      - numpy: NumPy library for performing mathematical operations.

    Example Usage:
    >>> score = estimate_blur_laplacian('image.jpg')
    >>> print(score)
    1724.1439391373214
    """

    # Read the image file in grayscale mode
    img = cv2.imread(image_file, cv2.COLOR_BGR2GRAY)

    # Apply the Laplacian operator to detect edges
    blur_map = cv2.Laplacian(img, cv2.CV_64F)

    # Calculate the variance of the edge map to estimate the blur score
    return np.var(blur_map)


def estimate_blur_svd(image_file, sv_num = 10):
    """
    Estimate the blur in an image file using Singular Value Decomposition (SVD).

    This function takes an image file path as input and calculates the blur score of the image using Singular Value Decomposition (SVD).
    The function first reads the image file using OpenCV and converts it to grayscale.
    The function then applies SVD to the grayscale image to obtain the singular values.
    The function then calculates the blur score by summing the top 'sv_num' singular values and dividing by the total number of singular values.

    Parameters:
    - image_file (str): The path of the image file to be processed.
    - sv_num (int): The number of top singular values to consider. Default value is 10.

    Return Value:
    - A float value representing the estimated blur score of the image.

    Dependencies:
    - The function requires the following libraries to be imported:
    - cv2: OpenCV library for reading image files and performing image processing operations.
    - numpy: NumPy library for performing mathematical operations.

    Example Usage:
    >>> score = estimate_blur_svd('image.jpg', sv_num=20)
    >>> print(score)
    0.5638945469402515
    """

    # Read the image file in grayscale mode
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Apply SVD to obtain the singular values
    u, s, v = np.linalg.svd(img)

    # Calculate the blur score by summing the top 'sv_num' singular values and dividing by the total number of singular values
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    score = top_sv / total_sv

    return score

