import imquality.brisque as brisque
from brisque import BRISQUE

def brisqueScoreTest(imagePath,brisqueThreshold):
    """
    Tests the BRISQUE score of an image against a given threshold.

    Parameters:
    imagePath (str): The path to the image to be tested.
    brisqueThreshold (float): The threshold BRISQUE score.

    Returns:
    bool: True if the BRISQUE score of the image is greater than the threshold, False otherwise.
    """
    brisquePredictor = BRISQUE(imagePath)
    brisqueScore = brisquePredictor.score()
    return brisqueScore>brisqueThreshold
