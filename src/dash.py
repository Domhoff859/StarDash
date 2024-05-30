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
    def __init__(self):
        pass
    
    def calculate(self, R, po_image, offset: float = 0.0) -> np.ndarray:
        """
        Calculates the dash representation of an Object Point Image.

        Args:
            R: rotation matrix
            po_image: an Object Point Image of the object

        Returns:
            po_dash: the dash representation of the po_image
        """
        # return np.einsum('bij,byxj->byxi', R, po_image) + offset
        return np.tensordot(po_image, R[0].T, axes=([3], [0])) + offset

    def _is_nan_or_inf(self, x: np.ndarray) -> bool:
        """
        Check if the input array contains NaN or Inf values.

        Args:
            x (np.ndarray): The input array

        Returns:
            bool: True if the input array contains NaN or Inf values, False otherwise
        """
        if np.isnan(x).sum() > 0 or np.isinf(x).sum():
            return True
        return False

    def _make_Rpxy(self, strides: int, shape, cam_K, coord_K):
        c = cam_K[:,:2,2]

        u, v = utils.generate_px_coordinates(shape, coord_K, strides)
        coords_c = np.stack(
            [
                u - c[:,0][:,np.newaxis,np.newaxis],
                v - c[:,1][:,np.newaxis,np.newaxis]
            ],
            axis=-1
        )

        f = np.stack([cam_K[:,0,0], cam_K[:,1,1]], axis=-1)
        # Reshape f to match the dimensions required for broadcasting
        f_reshaped = f[:, np.newaxis, np.newaxis]
        # Concatenate along the last axis
        coords_3d_with_z1 = np.concatenate(
            [
                coords_c / f_reshaped,
                np.ones_like(coords_c[:, :, :, :1])
            ],
            axis=-1
        )
        if self._is_nan_or_inf(coords_3d_with_z1):
            raise ValueError("coords_3d_with_z1 is not finite")

        z = np.constant([0,0,1], dtype=coords_3d_with_z1.dtype)

        axes = np.cross(z * np.ones_like(coords_3d_with_z1), coords_3d_with_z1)
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True) + 0.000001
        if self._is_nan_or_inf(axes):
            raise ValueError("axes is not finite")

        angles = utils.angle_between(z, coords_3d_with_z1)
        if self._is_nan_or_inf(angles):
            raise ValueError("angles is not finite")

        RpxRpy = utils.rot_matrix_from_angle(angles, axes)   
        if self._is_nan_or_inf(RpxRpy):
            raise ValueError("RpxRpy is not finite")
        return RpxRpy

    def remove_cam_effect(self, strides, v_cam, cam_K, coord_K):
        RpxRpy = self._make_Rpxy(strides, v_cam.shape[1:3], cam_K, coord_K)
        return np.einsum('bxyij, bxyj-> bxyi', RpxRpy, v_cam)

