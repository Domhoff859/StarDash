import numpy as np
import src.utils as utils

class DashRepresentation:
    """
    From the paper:
        "Handling Object Symmetries in CNN-based Pose Estimation"
        J. Richter-Klug and U. Frese
        2021 IEEE International Conference on Robotics and Automation (ICRA)
        Xi'an, China, 2021, pp. 13850-13856, doi: 10.1109/ICRA48506.2021.9561237

    The Dash representation:
    ========================
    The ambiguity of the star representation causes ignorance whether two points, whose values are close, also lie close on
    the object or e.g. on opposing ends. We use the pixelwise object points rotated into the camera. This is minus the vector from the
    object point to the object's origin relative to the camera. Note, this information is innately symmetrical invariant and (since we only rotated
    the object points) all angles between any object points are preserved, but no information regarding the object's rotation itself.
    The selected information can not be learned as is, since orientation is not a translation invariant function of the image
    Thus, depending on the pixel position in the image, we rotate the vector, such that the CNN can treat it as if in the image center.

        po_ij' = inv(Rray(i,j)) * Rc * po_ij

    Rray(i,j) is a matrix rotating the Z-axis onto the viewing ray of pixel (i, j). The viewing rays are defined by the camera calibration.
    Note that before this representation's usage the rotational offset must be reversed.
    """
    def __init__(self, model_info: dict) -> None:
        self.model_info = model_info
    
    def calculate(self, object_id: str, R: np.ndarray, po_image: np.ndarray) -> np.ndarray:
        """
        Calculates the dash representation of an Object Point Image.

        Args:
            R: rotation matrix
            po_image: an Object Point Image of the object

        Returns:
            po_dash: the dash representation of the po_image
        """
        offset = self.model_info[object_id]["symmetries_discrete"][0][:3,-1] / 2. if len(self.model_info[object_id]["symmetries_discrete"]) > 0 else 0 
        
        return np.einsum('bij,byxj->byxi', R, po_image) + offset
