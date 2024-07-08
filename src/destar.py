import logging
from math import isclose
import numpy as np

from . import utils

logger = logging.getLogger("Destar")

class DestarRepresentation:
    def __init__(self, model_info: dict, num_instances: int = None) -> None:
        """
        Initialize the DestarRepresentation object.

        Args:
            model_info (dict): Information about the model.
            num_instances (int, optional): Number of instances. Defaults to None.
        """
        # Set the model info
        self.model_info = model_info
        
        # Set the number of instances
        self.num_instances = num_instances if num_instances is not None else 1
        
    def calculate(self, star: np.ndarray, dash: np.ndarray, isvalid: np.ndarray, train_R: np.ndarray = None, object_id: str = None) -> np.ndarray:
        """
        Calculate the Destar representation.

        Args:
            object_id (str): ID of the object.
            star (np.ndarray): Star array.
            dash (np.ndarray): Dash array.
            isvalid (np.ndarray): Validity array.
            train_R (np.ndarray, optional): Training R array. Defaults to None.

        Returns:
            np.ndarray: Calculated Destar representation.
        """
        if object_id is None:
            model_info = self.model_info
        else:
            model_info = self.model_info[object_id]
        
        if model_info["symmetries_continuous"]:
            logger.debug("Destarring as symmetries_continuous")
            return self.best_continues_po(star, np.array([0,0,1], np.float32), train_R, star, dash, isvalid)

        if len(model_info["symmetries_discrete"]) == 0:
            logger.debug("Destarring is not changing anything")
            return star

        if isclose(model_info["symmetries_discrete"][0][2,2], 1, abs_tol=1e-3):
            factor = len(model_info["symmetries_discrete"])+1
            logger.debug(f"Destarring as symmetries_discrete with z_factor= {factor}")
            po_ = self.best_symmetrical_po(self.get_obj_star0_from_obj_star(star, z_factor=factor), factor, np.array([0,0,1], np.float32), train_R, model_info, star, dash, isvalid)

            offset = model_info["symmetries_discrete"][0][:3,-1] / 2.
            logger.debug(f"Po was corrected by {-offset}")
            return po_ - offset

        if isclose(model_info["symmetries_discrete"][0][1,1], 1, abs_tol=1e-3):
            factor = len(model_info["symmetries_discrete"])+1
            logger.debug(f"Destarring as symmetries_discrete with y_factor= {factor}")
            po_ = self.best_symmetrical_po(self.get_obj_star0_from_obj_star(star, y_factor=factor), factor, np.array([0,1,0], np.float32), train_R, model_info, star, dash, isvalid)

            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            logger.debug(f"Po was corrected by {-offset}")
            return po_
        
        assert(False)        
        
    def angle_substraction(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the angle subtraction between two arrays.

        Args:
            a (np.ndarray): First array.
            b (np.ndarray): Second array.

        Returns:
            np.ndarray: Angle subtraction result.
        """
        diff = (a - b) % (2 * np.pi)
        return np.minimum(diff, 2 * np.pi - diff)
    
    def best_symmetrical_po(self, star0: np.ndarray, factor: int, axis: np.ndarray, train_R: np.ndarray, model_info: dict, postar: np.ndarray, vo: np.ndarray, iseg: np.ndarray) -> np.ndarray:
        """
        Calculate the best symmetrical Po.

        Args:
            star0 (np.ndarray): Star0 array.
            factor (int): Symmetry factor.
            axis (np.ndarray): Axis array.
            train_R (np.ndarray): Training R array.
            model_info (dict): Model information.
            postar (np.ndarray): Postar array.
            vo (np.ndarray): Vo array.
            iseg (np.ndarray): Iseg array.

        Returns:
            np.ndarray: Best symmetrical Po.
        """
        if train_R is None:
            ref = self.generate_ref_data(model_info, postar, vo, iseg, 15, 3)
        else:
            ref = {
                'po': utils.eye(3, batch_shape=train_R.shape[:1])[:, np.newaxis],
                'vo': np.transpose(train_R[:, np.newaxis], (0, 1, 3, 2))
            }
        dash_angles = utils.angle_between(vo, ref['vo'], 'byxi, boji->byxoj')

        symR = utils.rot_matrix_from_angle(2 * np.pi / factor, axis)

        allp_pos = star0[..., np.newaxis, :]
        for _ in range(factor - 1):
            newp = np.einsum('ij, byxj -> byxi', symR, allp_pos[..., -1, :])
            allp_pos = np.concatenate([allp_pos, newp[..., np.newaxis, :]], axis=-2)

        allp_po_angles = utils.angle_between(allp_pos, ref['po'], 'byxsi, boji->byxosj')
        allp_angle_diffs = np.sum(self.angle_substraction(allp_po_angles, dash_angles[..., np.newaxis, :]) ** 2, axis=-1)

        arg_min: np.ndarray = np.argmin(allp_angle_diffs, axis=-1)
        arg_min_expanded = arg_min[..., np.newaxis]
        result_shape = arg_min.shape + (allp_pos.shape[-1],)
        best_po = np.empty(result_shape, dtype=allp_pos.dtype)
        
        batch_dims = 3
        batch_shape = allp_pos.shape[:batch_dims]
        for idx in np.ndindex(batch_shape):
            best_po[idx] = allp_pos[idx + (arg_min_expanded[idx],)]

        o_wide_error = np.sum(np.min(allp_angle_diffs, axis=-1)[..., np.newaxis] * iseg[..., np.newaxis], keepdims=True)
        arg_min = np.argmin(o_wide_error, axis=-1)
        arg_min_ = arg_min * iseg
        arg_min_expanded_ = np.array(arg_min_[..., np.newaxis], np.int8)
        result_shape = arg_min_.shape + (best_po.shape[-1],)
        best_po2 = np.empty(result_shape, dtype=best_po.dtype)
        
        batch_dims = 3
        batch_shape = best_po.shape[:batch_dims]
        for idx in np.ndindex(batch_shape):
            best_po2[idx] = best_po[idx + (arg_min_expanded_[idx],)]

        return np.sum(best_po2 * iseg[..., np.newaxis], axis=-2)
    
    def best_continues_po(self, star0: np.ndarray, axis: np.ndarray, train_R: np.ndarray, postar: np.ndarray, vo: np.ndarray, iseg: np.ndarray) -> np.ndarray:
        """
        Calculate the best continuous Po.

        Args:
            star0 (np.ndarray): Star0 array.
            axis (np.ndarray): Axis array.
            train_R (np.ndarray): Training R array.
            postar (np.ndarray): Postar array.
            vo (np.ndarray): Vo array.
            iseg (np.ndarray): Iseg array.

        Returns:
            np.ndarray: Best continuous Po.
        """
        if train_R is None:
            _R = self.make_oracle_R(self.direct_calc_z_dir(postar, vo, iseg))
        else:
            _R = train_R

        ref = {
            'po': utils.eye(3, batch_shape=_R.shape[:1])[:, np.newaxis],
            'vo': np.transpose(_R[:, np.newaxis], (0, 1, 3, 2))
        }

        dash_angles = utils.angle_between(vo, ref['vo'], 'byxi, boji->byxoj')
        star0_ = star0[..., np.newaxis, np.newaxis, :] * np.ones_like(ref['po'])[:, np.newaxis, np.newaxis]
        star0_under_ref = utils.change_angle_around_axis(
            axis * np.ones_like(star0_), star0_, ref['po'][:, np.newaxis, np.newaxis] * np.ones_like(star0_), 0, 'byxoji ,byxoji ->byxoj')

        po_star0_angles = utils.angle_between(star0_under_ref, ref['po'], 'byxoji, boji->byxoj')

        beta_upper_part = np.cos(dash_angles)
        beta_lower_part = np.cos(po_star0_angles)

        quotient = beta_upper_part / beta_lower_part
        quotient = np.clip(quotient, -0.9999, 0.9999)
        beta = np.arccos(quotient)

        R_betas = utils.rot_matrix_from_angle(np.stack([beta, -beta], axis=-1), axis)

        allp_pos = np.einsum('byxojaki,byxoji->byxojak', R_betas, star0_under_ref)
        allp_pos = np.concatenate([allp_pos[..., 0, :], allp_pos[..., 1, :]], axis=-2)
        
        allp_po_angles = utils.angle_between(allp_pos, ref['po'], 'byxosi, boji->byxosj')
        allp_angle_diffs = np.sum(self.angle_substraction(allp_po_angles, dash_angles[..., np.newaxis, :]) ** 2, axis=-1)
        
        arg_min: np.ndarray = np.argmin(allp_angle_diffs, axis=-1)
        arg_min_expanded = arg_min[..., np.newaxis]
        result_shape = arg_min.shape + (allp_pos.shape[-1],)
        best_po = np.empty(result_shape, dtype=allp_pos.dtype)
        
        batch_dims = 4
        batch_shape = allp_pos.shape[:batch_dims]
        for idx in np.ndindex(batch_shape):
            best_po[idx] = allp_pos[idx + (arg_min_expanded[idx],)]

        o_wide_error = np.sum(np.array(np.min(allp_angle_diffs, axis=-1)[..., np.newaxis,:] * iseg[..., np.newaxis]), keepdims=True)
        arg_min = np.argmin(o_wide_error, axis=-1)
        arg_min_ = arg_min * np.array(iseg, dtype=arg_min.dtype)
        arg_min_expanded_ = arg_min_[..., np.newaxis]
        result_shape = arg_min_.shape + (best_po.shape[-1],)
        best_po2 = np.empty(result_shape, dtype=best_po.dtype)
        
        batch_dims = 3
        batch_shape = best_po.shape[:batch_dims]
        for idx in np.ndindex(batch_shape):
            best_po2[idx] = best_po[idx + (arg_min_expanded_[idx],)]

        return np.sum(best_po2 * iseg[..., np.newaxis], axis=-2)
    
    def get_obj_star0_from_obj_star(self, obj_star: np.ndarray, x_factor: int = 1, y_factor: int = 1, z_factor: int = 1) -> np.ndarray:
        """
        Get the object star0 from object star.

        Args:
            obj_star (np.ndarray): Object star array.
            x_factor (int, optional): X factor. Defaults to 1.
            y_factor (int, optional): Y factor. Defaults to 1.
            z_factor (int, optional): Z factor. Defaults to 1.

        Returns:
            np.ndarray: Object star0 array.
        """
        R = utils.eye(3, batch_shape=obj_star.shape[:-1])
        obj_star = utils.change_angle_around_axis(R[..., 2], obj_star, R[..., 0], 1. / z_factor)
        obj_star = utils.change_angle_around_axis(R[..., 1], obj_star, R[..., 2], 1. / y_factor)
        obj_star = utils.change_angle_around_axis(R[..., 0], obj_star, R[..., 1], 1. / x_factor)
        return obj_star

    def direct_calc_z_dir(self, postar: np.ndarray, vo_image: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """
        Calculate the Z direction using direct calculation.

        Args:
            postar (np.ndarray): Postar array.
            vo_image (np.ndarray): Vo image array.
            seg (np.ndarray): Seg array.

        Returns:
            np.ndarray: Calculated Z direction.
        """
        # Create A and b matrices
        A = (vo_image * seg)[..., np.newaxis,:]
        AtA = np.matmul(A.T, A)
        b = (postar * seg)[...,2:,np.newaxis]
        Atb = np.matmul(A.T, b)
        
        _AtA = np.sum(AtA, axis=(1, 2))
        _Atb = np.sum(Atb, axis=(1, 2))
        return np.matmul(np.linalg.pinv(_AtA), _Atb).transpose(2, 0, 1)

    def make_oracle_R(self, Rz: np.ndarray) -> np.ndarray:
        """
        Make the oracle R matrix.

        Args:
            Rz (np.ndarray): Rz array.

        Returns:
            np.ndarray: Oracle R matrix.
        """
        z = utils.normalize(Rz[..., 0])
        o = np.concatenate([z[..., 1:], z[..., :1]], axis=-1)
        x = np.cross(z, o)
        y = np.cross(z, x)
        z = np.cross(x, y)
        return np.stack([x, y, z], axis=-1)

    def _make_ref_outof_samples_symmetrical(self, vo_samples: np.ndarray, star0_samples: np.ndarray, factor: int, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make the reference out of samples for symmetrical calculations.

        Args:
            vo_samples (np.ndarray): Vo samples array.
            star0_samples (np.ndarray): Star0 samples array.
            factor (int): Symmetry factor.
            axis (np.ndarray): Axis array.

        Returns:
            tuple[np.ndarray, np.ndarray]: Reference Vo and Po arrays.
        """
        ref_vo = vo_samples
        dash_angles = utils.angle_between(vo_samples, ref_vo, dot_product='bski, bsji->bsjk')

        symR = utils.rot_matrix_from_angle(2 * np.pi / factor, axis)

        allp_pos = star0_samples[..., np.newaxis, :]

        for _ in range(factor - 1):
            newp = np.einsum('ij, bskj -> bski', symR, allp_pos[:, :, -1, :])
            allp_pos = np.concatenate([allp_pos, newp[..., np.newaxis, :]], axis=-2)

        sample_size = 3
        meshgrid = np.meshgrid(np.arange(factor), np.arange(factor), np.arange(factor))
        gather_per = np.array(meshgrid).reshape((sample_size, -1))
        gather_per = gather_per[..., np.newaxis] * np.ones_like(allp_pos[:, :, :, :1, :1], dtype=gather_per.dtype)
        gather_per_rev = np.array(meshgrid).reshape((-1, sample_size))

        all_combi = allp_pos[gather_per[:, 0], gather_per[:, 1], gather_per[:, 2]]

        all_combi_po_angles = utils.angle_between(all_combi, all_combi, dot_product='bskni, bsjni->bsjkn')

        all_combi_angle_diffs = np.sum(self.angle_substraction(all_combi_po_angles, dash_angles[..., np.newaxis]), axis=[-2, -3])

        arg_min = np.argmin(all_combi_angle_diffs, axis=-1)

        arg_min_combi = gather_per_rev[arg_min]

        best_pos = allp_pos[arg_min_combi[:, 0], arg_min_combi[:, 1], arg_min_combi[:, 2]]

        ref_po = best_pos

        return (ref_vo, ref_po)

    def _generate_samples_per_batch(self, sb_postar: np.ndarray, sb_vo: np.ndarray, sb_iseg: np.ndarray, counts: int, sample_size: int) -> np.ndarray:
        """
        Generate samples per batch.

        Args:
            sb_postar (np.ndarray): SB postar array.
            sb_vo (np.ndarray): SB vo array.
            sb_iseg (np.ndarray): SB iseg array.
            counts (int): Number of counts.
            sample_size (int): Sample size.

        Returns:
            np.ndarray: Generated samples.
        """
        samples = []
        for i in range(self.num_instances):
            si_seg = sb_iseg[:, :, i]
            selection_index = np.argwhere(si_seg > 0.5)
            if selection_index.size > 0:
                vo_sel = sb_vo[selection_index[:, 0], selection_index[:, 1]]
                postar_sel = sb_postar[selection_index[:, 0], selection_index[:, 1]]

                pos = np.random.randint(0, vo_sel.shape[0], size=(counts, sample_size))

                vo_samples = vo_sel[pos]
                postar_samples = postar_sel[pos]
            else:
                vo_samples = np.ones((counts, sample_size, 3))
                postar_samples = np.ones((counts, sample_size, 3))

            samples.append(np.stack([vo_samples, postar_samples], axis=-1))
        return np.concatenate(samples, axis=0)

    def generate_ref_data(self, model_info: dict, postar: np.ndarray, vo: np.ndarray, iseg: np.ndarray, counts: int, sample_size: int) -> dict[str, np.ndarray]:
        """
        Generate reference data.

        Args:
            model_info (dict): Model information.
            postar (np.ndarray): Postar array.
            vo (np.ndarray): Vo array.
            iseg (np.ndarray): Iseg array.
            counts (int): Number of counts.
            sample_size (int): Sample size.

        Returns:
            dict[str, np.ndarray]: Generated reference data.
        """
        samples = np.array(
            [
                self._generate_samples_per_batch(
                    *args,
                    counts=counts,
                    sample_size=sample_size,
                ) 
                for args in zip(postar, vo, iseg)
            ], dtype=np.float32
        )

        vo_samples = samples[...,0]
        postar_samples = samples[...,1]

        if "symmetries_continuous" in model_info:
            print("generate ref samples for continuous symmetries around z")
            #This code does not work
            ref_vo, ref_po = self._make_ref_outof_samples_symmetrical(vo_samples, np.array([0, 0, 1], dtype=np.float32))

        elif "symmetries_discrete" in model_info:
            sym_discrete = model_info["symmetries_discrete"]
            if isclose(sym_discrete[0][2,2], 1, abs_tol=1e-3):
                factor = len(sym_discrete)+1
                print("generate ref samples discrete symmetries with z_factor=", factor)
                ref_vo, ref_po = self._make_ref_outof_samples_symmetrical(
                    vo_samples,
                    self.get_obj_star0_from_obj_star(postar_samples, z_factor=factor),
                    factor,
                    np.array([0, 0, 1], dtype=np.float32)
                )

            elif isclose(sym_discrete[0][1,1], 1, abs_tol=1e-3):
                factor = len(sym_discrete)+1
                print("generate ref samples discrete symmetries with y_factor=", factor)
                ref_vo, ref_po = self._make_ref_outof_samples_symmetrical(
                    vo_samples,
                    self.get_obj_star0_from_obj_star(postar_samples, y_factor=factor),
                    factor,
                    np.array([0, 0, 1], dtype=np.float32)
                )
        else:
            raise ValueError("Something went wrong.")
        return { 'po': ref_po, 'vo': ref_vo }
