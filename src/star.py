import numpy as np
import math

import src.utils as utils

class StarRepresentation:
    """
    From the paper:
        "Handling Object Symmetries in CNN-based Pose Estimation"
        J. Richter-Klug and U. Frese
        2021 IEEE International Conference on Robotics and Automation (ICRA)
        Xi'an, China, 2021, pp. 13850-13856, doi: 10.1109/ICRA48506.2021.9561237

    The Star representation:
    ========================
    The representation is a modification of the object points such that rotating by one step of symmetry, i.e. 2π/n , is a
    simple closed curve in the representation. In it, all object points, that appear the same (based on the
    defined symmetry), are mapped on the same value and no possible rotation will result in an uncontinuous change.
    Therefore, the representation becomes symmetry aware, but also ambiguous. To gain the star representation of the object
    points, these are first transformed in cylindrical coordinate space, where the cylindric axis is aligned with the symmetry axis.
    Here the angle value is multiplied by n (the fold of symmetry). Afterwards the points are transformed back to Cartesian vector space.

                            [n    ]
        po_ij* = cartetsian( [  1  ] cylindric(po_ij) )      Note: This assumes Z as symmetry axis.
                            [    1]

    Let's have a closer look at the folds of symmetry extremes:
    On the lower end, one finds non-symmetrical objects (n = 1); In this case the star representation is identical to the
    origin object points which is the expected outcome. On the other end, we find objects with infinity-fold symmetries, e.g.
    bottles. Here an infinitely small step of rotation closes one step of symmetry. Since the multiplication with infinity is
    unhandy, in this case, we multiply the angle values with zero. Therefore, all points have the same angle around the
    rotation axis as they all are equivalent under symmetry.
    """
    def collapses_obj_to_dot_symmetry(self, obj, x_factor=1, y_factor=1, z_factor=1):
        R = utils.eye(3, batch_shape=obj.shape[:-1])
        obj = utils.change_angle_around_axis(R[...,0], obj, R[...,1], x_factor)
        obj = utils.change_angle_around_axis(R[...,1], obj, R[...,2], y_factor)
        obj = utils.change_angle_around_axis(R[...,2], obj, R[...,0], z_factor)
        return obj

    def calculate(self, model_info, po_image):
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
            return self.collapses_obj_to_dot_symmetry(po_image, z_factor=0)

        if "symmetries_discrete" not in model_info:
            #Starring does not change anything - no symmetries available
            return po_image
        else:
            sym_discrete = model_info["symmetries_discrete"]

            if math.isclose(sym_discrete[0][2, 2], 1, abs_tol=1e-3):
                #Object correction by offset
                offset = sym_discrete[0][:3, -1] / 2.0
                po_image = po_image + offset
                return self.collapses_obj_to_dot_symmetry(po_image, z_factor=len(sym_discrete)+1)

            if math.isclose(sym_discrete[0][1, 1], 1, abs_tol=1e-3):
                offset = sym_discrete[0][:3, -1] / 2.0
                po_image = po_image + offset
                return self.collapses_obj_to_dot_symmetry(po_image, y_factor=len(sym_discrete)+1)