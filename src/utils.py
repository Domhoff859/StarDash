import numpy as np

def eye(num_rows, num_columns=None, batch_shape=None, dtype=float):
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

def normalize(x):
    mag = np.linalg.norm(x, axis=-1, keepdims=True) + 0.00001
    return x / mag

def cross_n(x,y):
    return normalize(np.cross(x,y))

def angle_between(x, y, dot_product='i, bxyi->bxy'):
    norm_x, norm_y = normalize(x), normalize(y)
    numerator = np.einsum(dot_product, norm_x, norm_y)
    # Define the clipping boundaries
    min_val = 0.00001 - 1.0
    max_val = 1.0 - 0.00001
    # Apply clipping to limit the values to the range [min_val, max_val]
    clipped_numerator = np.clip(numerator, min_val, max_val)
    # Compute the arccosine of the clipped values
    return np.arccos(clipped_numerator)

def get_angle_around_axis(axis, v_from, v_to, dot_product='bxyi, bxyi->bxy'):
    corrected_v_from = cross_n(np.cross(axis, v_from), axis)
    corrected_v_to = cross_n(np.cross(axis, v_to), axis)
    
    angle = angle_between(corrected_v_from, corrected_v_to, dot_product=dot_product)
    
    new_axis = cross_n(corrected_v_from, corrected_v_to)
    n = np.linalg.norm(new_axis + axis, axis=-1, keepdims=True)
    #NumPy does not have an equivalent concept of gradient tracking or stopping gradients like TensorFlow does.
    sign_correction_factor = np.squeeze(np.math.sign(n - 1.0), axis=-1)
    angle *= np.minimum(sign_correction_factor * 2.0 + 1, 1)
    return angle


def change_angle_around_axis(axis, x, v_zero, factor, dot_product='bxyi, bxyi->bxy'):
    factor = factor if not np.isinf(factor) else 0
    current_angle = get_angle_around_axis(axis, v_zero, x, dot_product=dot_product)# + np.pi
    angle_change = current_angle * (factor - 1) 
    R = rot_matrix_from_angle(angle_change, axis)
    return np.squeeze(R @ np.expand_dims(x, axis=-1), axis=-1)

def rot_matrix_from_angle(angle, axis):
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


def generate_px_coordinates(shape, coord_K, strides=1):
    x_range = np.arange(shape[1], dtype=np.float32)
    y_range = np.arange(shape[0], dtype=np.float32)
    # Generate the meshgrid
    u, v = np.meshgrid(x_range, y_range)
    return (
        u * coord_K[:,0:1,0:1] * strides + coord_K[:,1:2,0:1],
        v * coord_K[:,0:1,1:2] * strides + coord_K[:,1:2,1:2]
    )

def depth_based_cam_coords(depth, cam_K, coord_K, strides):
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

def cam_to_obj(R, t, cam_coords):
    # Expand dimensions of t to match the shape of cam_coords
    t_expanded = t[:, np.newaxis, np.newaxis, :]
    # Perform matrix multiplication and subtraction
    # Old: np.einsum('bji,byxj->byxi', R, cam_coords - t[:, np.newaxis, np.newaxis])
    return R @ (cam_coords - t_expanded)

def obj_validity(image, model_info):
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

def dataset_conversion_layers(xDim, yDim, model_info, strides=1):
    # Generate empty input tensors
    inputs = {
        'rgb': np.zeros((1, yDim, xDim, 3)),
        'depth': np.zeros((1, yDim, xDim)),
        'segmentation': np.zeros((1, yDim, xDim), dtype=np.int32),
        'camera_matrix': np.zeros((1, 3, 3)),
        'coord_offset': np.zeros((1, 2, 2)),
        'rotationmatrix': np.zeros((1, 3, 3)),
        'translation': np.zeros((1, 3)),
    }
    
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