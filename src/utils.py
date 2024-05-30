import numpy as np
from typing import Tuple, Optional, Type, Dict

def eye(num_rows: int, num_columns: Optional[int] = None, batch_shape: Optional[Tuple[int]] = None, dtype: Type[float] = float) -> np.ndarray:
    """
    Create a NumPy identity matrix or a batch of identity matrices.
    
    Parameters:
        num_rows (int): Number of rows in each identity matrix.
        num_columns (int, optional): Number of columns in each identity matrix. Defaults to num_rows.
        batch_shape (tuple of int, optional): Shape of the batch of identity matrices. Defaults to None.
        dtype (data-type, optional): Data type of the output array. Defaults to float.
        
    Returns:
        np.ndarray: Identity matrix or batch of identity matrices.
    """
    if num_columns is None:
        num_columns = num_rows
    
    # Create the core identity matrix
    eye_matrix = np.eye(num_rows, num_columns, dtype=dtype)
    
    if batch_shape is not None:
        # Expand the eye matrix to the batch shape
        result_shape = batch_shape + eye_matrix.shape
        return np.broadcast_to(eye_matrix, result_shape)
    else:
        return eye_matrix

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize a vector or an array of vectors.
    
    Parameters:
        x (np.ndarray): Input vector or array of vectors.
        
    Returns:
        np.ndarray: Normalized vector or array of vectors.
    """
    mag = np.linalg.norm(x, axis=-1, keepdims=True) + 0.00001
    return x / mag

def cross_n(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the cross product of two vectors or arrays of vectors and normalize the result.
    
    Parameters:
        x (np.ndarray): First vector or array of vectors.
        y (np.ndarray): Second vector or array of vectors.
        
    Returns:
        np.ndarray: Cross product of the input vectors or arrays of vectors, normalized.
    """
    return normalize(np.cross(x, y))

def angle_between(x: np.ndarray, y: np.ndarray, dot_product: str = 'i, bxyi->bxy') -> np.ndarray:
    """
    Compute the angle between two vectors or arrays of vectors.
    
    Parameters:
        x (np.ndarray): First vector or array of vectors.
        y (np.ndarray): Second vector or array of vectors.
        dot_product (str, optional): Dot product equation. Defaults to 'i, bxyi->bxy'.
        
    Returns:
        np.ndarray: Angle between the input vectors or arrays of vectors.
    """
    norm_x, norm_y = normalize(x), normalize(y)
    numerator = np.einsum(dot_product, norm_x, norm_y)
    # Define the clipping boundaries
    min_val = 0.00001 - 1.0
    max_val = 1.0 - 0.00001
    # Apply clipping to limit the values to the range [min_val, max_val]
    clipped_numerator = np.clip(numerator, min_val, max_val)
    # Compute the arccosine of the clipped values
    return np.arccos(clipped_numerator)

def get_angle_around_axis(axis: np.ndarray, v_from: np.ndarray, v_to: np.ndarray, dot_product: str = 'bxyi, bxyi->bxy') -> np.ndarray:
    """
    Compute the angle around an axis between two vectors or arrays of vectors.
    
    Parameters:
        axis (np.ndarray): Axis vector or array of axis vectors.
        v_from (np.ndarray): First vector or array of vectors.
        v_to (np.ndarray): Second vector or array of vectors.
        dot_product (str, optional): Dot product equation. Defaults to 'bxyi, bxyi->bxy'.
        
    Returns:
        np.ndarray: Angle around the axis between the input vectors or arrays of vectors.
    """
    corrected_v_from = cross_n(np.cross(axis, v_from), axis)
    corrected_v_to = cross_n(np.cross(axis, v_to), axis)
    
    angle = angle_between(corrected_v_from, corrected_v_to, dot_product=dot_product)
    
    new_axis = cross_n(corrected_v_from, corrected_v_to)
    n = np.linalg.norm(new_axis + axis, axis=-1, keepdims=True)
    #NumPy does not have an equivalent concept of gradient tracking or stopping gradients like TensorFlow does.
    sign_correction_factor = np.squeeze(np.math.sign(n - 1.0), axis=-1)
    angle *= np.minimum(sign_correction_factor * 2.0 + 1, 1)
    return angle


def change_angle_around_axis(axis: np.ndarray, x: np.ndarray, v_zero: np.ndarray, factor: float, dot_product: str = 'bxyi, bxyi->bxy') -> np.ndarray:
    """
    Change the angle around an axis of a vector or an array of vectors.
    
    Parameters:
        axis (np.ndarray): Axis vector or array of axis vectors.
        x (np.ndarray): Input vector or array of vectors.
        v_zero (np.ndarray): Zero vector or array of zero vectors.
        factor (float): Factor to change the angle by.
        dot_product (str, optional): Dot product equation. Defaults to 'bxyi, bxyi->bxy'.
        
    Returns:
        np.ndarray: Vector or array of vectors with the changed angle around the axis.
    """
    factor = factor if not np.isinf(factor) else 0
    current_angle = get_angle_around_axis(axis, v_zero, x, dot_product=dot_product)# + np.pi
    angle_change = current_angle * (factor - 1) 
    R = rot_matrix_from_angle(angle_change, axis)
    return np.squeeze(R @ np.expand_dims(x, axis=-1), axis=-1)

def rot_matrix_from_angle(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Create a rotation matrix from an angle and an axis.
    
    Parameters:
        angle (np.ndarray): Angle or array of angles.
        axis (np.ndarray): Axis vector or array of axis vectors.
        
    Returns:
        np.ndarray: Rotation matrix or array of rotation matrices.
    """
    #See: https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c

    part_one = c[..., np.newaxis, np.newaxis] * np.eye(3, dtype=c.dtype)
    # We use np.einsum with the ellipsis notation (...)
    # to perform element-wise multiplication of the axis array with itself.
    # Resulting in:
    # [x*x, x*y, x*z]
    # [x*y, y*y, y*z]
    # [x*z, y*z, z*z]
    part_two = t[..., np.newaxis, np.newaxis] * np.einsum('...i,...j->...ij', axis, axis)

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    part_three = s[..., np.newaxis, np.newaxis] * np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    return part_one + part_two - part_three


def generate_px_coordinates(shape: Tuple[int], coord_K: np.ndarray, strides: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pixel coordinates based on the shape and camera intrinsics.
    
    Parameters:
        shape (Tuple[int]): Shape of the image.
        coord_K (np.ndarray): Camera intrinsics.
        strides (int, optional): Strides for subsampling. Defaults to 1.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated pixel coordinates.
    """
    x_range = np.arange(shape[1], dtype=np.float32)
    y_range = np.arange(shape[0], dtype=np.float32)
    # Generate the meshgrid
    u, v = np.meshgrid(x_range, y_range)
    return (
        u * coord_K[:,0:1,0:1] * strides + coord_K[:,1:2,0:1],
        v * coord_K[:,0:1,1:2] * strides + coord_K[:,1:2,1:2]
    )

def depth_based_cam_coords(depth: np.ndarray, cam_K: np.ndarray, coord_K: np.ndarray, strides: int) -> np.ndarray:
    """
    Compute camera coordinates based on depth and camera intrinsics.
    
    Parameters:
        depth (np.ndarray): Depth map.
        cam_K (np.ndarray): Camera intrinsics.
        coord_K (np.ndarray): Coordinate intrinsics.
        strides (int): Strides for subsampling.
        
    Returns:
        np.ndarray: Camera coordinates.
    """
    # Create coordinate grids using broadcasting
    u = np.arange(depth.shape[2], dtype=np.float32) * coord_K[:, 0, 0] * strides + coord_K[:, 1, 0]
    v = np.arange(depth.shape[1], dtype=np.float32) * coord_K[:, 0, 1] * strides + coord_K[:, 1, 1]
    
    # Reshape u and v to match the shape of depth
    u = u[:, np.newaxis, np.newaxis]
    v = v[:, np.newaxis, np.newaxis]
    
    # Compute scaled coordinates
    scaled_coords = np.stack([u * depth, v * depth, depth], axis=-1)
    
    # Perform matrix multiplication using broadcasting
    inv_cam_K = np.linalg.inv(cam_K)
    return inv_cam_K @ scaled_coords

def cam_to_obj(R: np.ndarray, t: np.ndarray, cam_coords: np.ndarray) -> np.ndarray:
    """
    Convert camera coordinates to object space.
    
    Parameters:
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.
        cam_coords (np.ndarray): Camera coordinates.
        
    Returns:
        np.ndarray: Object space coordinates.
    """
    # Expand dimensions of t to match the shape of cam_coords
    t_expanded = t[:, np.newaxis, np.newaxis, :]
    # Perform matrix multiplication and subtraction
    return R @ (cam_coords - t_expanded)

def obj_validity(image: np.ndarray, model_info: Dict[str, float]) -> np.ndarray:
    """
    Compute the validity of object space coordinates.
    
    Parameters:
        image (np.ndarray): Image.
        model_info (Dict[str, float]): Model information.
        
    Returns:
        np.ndarray: Validity mask.
    """
    obj_mins = 1.1 * np.array(
        [
            model_info["min_x"],
            model_info["min_y"],
            model_info["min_z"]
        ],
        dtype=np.float32
    )
    obj_maxs = 1.1 * np.array(
        [
            model_info["min_x"] + model_info["size_x"],
            model_info["min_y"] + model_info["size_y"],
            model_info["min_z"] + model_info["size_z"]
        ],
        dtype=np.float32
    )
    # Perform element-wise comparisons
    is_in_range = np.logical_and(np.less(obj_mins, image), np.less(image, obj_maxs))
    # Check if all dimensions are in range
    is_valid = np.all(is_in_range, axis=-1, keepdims=True)
    # Convert boolean array to float32
    return is_valid.astype(np.float32)

def dataset_conversion_layers(inputs: Dict[str, np.ndarray], xDim: int, yDim: int, model_info: Dict[str, float], strides: int = 1) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert input data to the desired format for dataset conversion.
    
    Parameters:
        inputs (Dict[str, np.ndarray]): Input data.
        xDim (int): Width of the image.
        yDim (int): Height of the image.
        model_info (Dict[str, float]): Model information.
        strides (int, optional): Strides for subsampling. Defaults to 1.
        
    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Converted data.
    """
    # Generate empty input tensors
    # inputs = {
    #     'rgb': np.zeros((1, yDim, xDim, 3)),
    #     'depth': np.zeros((1, yDim, xDim)),
    #     'segmentation': np.zeros((1, yDim, xDim), dtype=np.int32),
    #     'camera_matrix': np.zeros((1, 3, 3)),
    #     'coord_offset': np.zeros((1, 2, 2)),
    #     'rotationmatrix': np.zeros((1, 3, 3)),
    #     'translation': np.zeros((1, 3)),
    # }
    
    # Subsample depth and segmentation
    depth = inputs['depth'][:, ::strides, ::strides]
    segmentations = inputs['segmentation'][:, ::strides, ::strides]
    # Convert segmentation to binary mask
    segmentations = segmentations[..., np.newaxis] > 0
    
    # Compute camera coordinates
    cam_coords = depth_based_cam_coords(depth, inputs['camera_matrix'], inputs['coord_offset'], strides)
    
    # Convert camera coordinates to object space
    obj_image = cam_to_obj(inputs['rotationmatrix'], inputs['translation'], cam_coords)
    
    # Compute object validity mask
    isvalid = obj_validity(obj_image)
    
    # Apply segmentation mask and object validity
    isvalid = isvalid * segmentations
    obj_image = obj_image * isvalid
    
    # Return inputs and processed data
    segmentation = inputs['segmentation'][:, ::strides, ::strides]
    return inputs, obj_image, isvalid, depth, segmentation
