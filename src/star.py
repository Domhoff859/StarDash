"""
From the paper:
    "Handling Object Symmetries in CNN-based Pose Estimation"
    J. Richter-Klug and U. Frese
    2021 IEEE International Conference on Robotics and Automation (ICRA)
    Xi'an, China, 2021, pp. 13850-13856, doi: 10.1109/ICRA48506.2021.9561237

The Star representation:
========================

"""
import json
import numpy as np
import math

import utils

def collapses_obj_to_dot_symmetry(obj, x_factor=1, y_factor=1, z_factor=1):
    R = utils.eye(3, batch_shape=obj.shape[:-1])
    obj = utils.change_Angle_around_Axis(R[...,0], obj, R[...,1], x_factor)
    obj = utils.change_Angle_around_Axis(R[...,1], obj, R[...,2], y_factor)
    obj = utils.change_Angle_around_Axis(R[...,2], obj, R[...,0], z_factor)
    return obj

def calculate(model_info, po_image):
    """
    Calculates the star representation of an Object Point Image.

    Args:
        model_info: the object's model info (incl. symmetries)
        po_image: an Object Point Image of the object

    Returns:
        po_star: the star representation of the po_image
    """
    if "symmetries_continuous" in model_info:
        print("Starring as symmetries_continuous")
        return collapses_obj_to_dot_symmetry(po_image, z_factor=0)

    if "symmetries_discrete" not in model_info:
        #Starring does not change anything - no symmetries available
        return po_image
    else:
        sym_discrete = model_info["symmetries_discrete"]

        if math.isclose(sym_discrete[0][2, 2], 1, abs_tol=1e-3):
            #Object correction by offset
            offset = sym_discrete[0][:3, -1] / 2.0
            po_image = po_image + offset
            return collapses_obj_to_dot_symmetry(po_image, z_factor=len(sym_discrete)+1)

        if math.isclose(sym_discrete[0][1, 1], 1, abs_tol=1e-3):
            offset = sym_discrete[0][:3, -1] / 2.0
            po_image = po_image + offset
            return collapses_obj_to_dot_symmetry(po_image, y_factor=len(sym_discrete)+1)