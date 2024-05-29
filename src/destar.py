import math
import numpy as np

import utils

def get_obj_star0_from_obj_star(obj_star, x_factor=1, y_factor=1, z_factor=1):
    R = utils.eye(3, batch_shape=obj_star.shape[:-1])
    obj_star = utils.change_angle_around_axis(R[2], obj_star, R[0], 1. / z_factor)
    obj_star = utils.change_angle_around_axis(R[1], obj_star, R[2], 1. / y_factor)
    obj_star = utils.change_angle_around_axis(R[0], obj_star, R[1], 1. / x_factor)
    return obj_star

def angle_substraction(a, b):
    diff = (a - b) % (2 * np.pi)
    return np.minimum(diff, 2 * np.pi - diff)

def direct_calc_z_dir(postar, vo_image, seg):
    # Create A and b matrices
    A = (vo_image * seg)[..., np.newaxis, :]
    b = (postar * seg)[..., 2:, np.newaxis]
    
    # Calculate pseudo-inverse directly
    _AtA = np.einsum('...ji,...jk->...ik', A, A)
    _Atb = np.einsum('...ji,...jk->...ik', A, b)
    return np.linalg.pinv(_AtA) @ _Atb

def make_oracle_R(Rz):
    z = utils.normalize(Rz[..., 0])
    o = np.concatenate([z[..., 1:], z[..., :1]], axis=-1)
    x = np.cross(z, o)
    y = np.cross(z, x)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def calculate(model_info, postar, vo, iseg, train_R=None, num_instances=1):

    def best_symmetrical_po(star0, factor, axis):
        if train_R is None:
            ref = generate_ref_data(model_info, postar, vo, iseg, 15, 3, num_instances=num_instances)
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
        allp_angle_diffs = np.sum(angle_substraction(allp_po_angles, dash_angles[..., np.newaxis, :]) ** 2, axis=-1)

        arg_min = np.argmin(allp_angle_diffs, axis=-1)
        best_po = np.take_along_axis(allp_pos, arg_min[..., np.newaxis], axis=-2)

        o_wide_error = np.sum(np.min(allp_angle_diffs, axis=-1)[..., np.newaxis] * iseg[..., np.newaxis], axis=[1, 2], keepdims=True)
        arg_min = np.argmin(o_wide_error, axis=-1)
        arg_min_ = arg_min * iseg

        best_po = np.take_along_axis(best_po, arg_min_[..., np.newaxis], axis=-2)
        return np.sum(best_po * iseg[..., np.newaxis], axis=-2)

    def best_continues_po(star0, axis):
        if train_R is None:
            _R = make_oracle_R(direct_calc_z_dir(postar, vo, iseg))
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
        quotient = np.minimum(quotient, 0.9999)
        quotient = np.maximum(quotient, -0.9999)
        beta = np.arccos(quotient)

        R_betas = utils.rot_matrix_from_angle(np.stack([beta, -beta], axis=-1), axis)

        allp_pos = np.einsum('byxojaki,byxoji->byxojak', R_betas, star0_under_ref)
        allp_pos = np.concatenate([allp_pos[..., 0, :], allp_pos[..., 1, :]], axis=-2)

        allp_po_angles = utils.angle_between(allp_pos, ref['po'], 'byxosi, boji->byxosj')
        allp_angle_diffs = np.sum(angle_substraction(allp_po_angles, dash_angles[..., np.newaxis, :]) ** 2, axis=-1)

        arg_min = np.argmin(allp_angle_diffs, axis=-1)
        best_po = np.take_along_axis(allp_pos, arg_min[..., np.newaxis], axis=-2)

        o_wide_error = np.sum(np.min(allp_angle_diffs, axis=-1)[..., np.newaxis] * iseg[..., np.newaxis], axis=[1, 2], keepdims=True)
        arg_min = np.argmin(o_wide_error, axis=-1)
        arg_min_ = arg_min * iseg

        best_po = np.take_along_axis(best_po, arg_min_[..., np.newaxis], axis=-2)
        return np.sum(best_po * iseg[..., np.newaxis], axis=-2)

    if "symmetries_continuous" in model_info:
        print("destarring as symmetries_continuous")
        return best_continues_po(postar, np.array([0, 0, 1], dtype=np.float32))

    if "symmetries_discrete" not in model_info:
        print("destarring is not changing anything")
        return postar
    else:
        sym_discrete = model_info["symmetries_discrete"]

        if math.isclose(sym_discrete[0][2, 2], 1, abs_tol=1e-3):
            factor = len(sym_discrete) + 1
            print("destarring as symmetries_discrete with z_factor=", factor)
            po_ = best_symmetrical_po(
                get_obj_star0_from_obj_star(postar, z_factor=factor),
                factor, np.array([0, 0, 1],
                dtype=np.float32)
            )

            offset = sym_discrete[0][:3, -1] / 2.0
            print("po_ was corrected by", -offset)
            return po_ - offset

        if math.isclose(sym_discrete[0][1, 1], 1, abs_tol=1e-3):
            factor = len(sym_discrete) + 1
            print("destarring as symmetries_discrete with y_factor=", factor)
            po_ = best_symmetrical_po(
                get_obj_star0_from_obj_star(postar, y_factor=factor),
                factor,
                np.array([0, 1, 0], dtype=np.float32)
            )

            offset = sym_discrete[0][:3, -1] / 2.0
            print("po_ was corrected by", -offset)
            return po_ - offset

def __make_ref_outof_samples_symmetrical(vo_samples, star0_samples, factor, axis):
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

    all_combi_angle_diffs = np.sum(angle_substraction(all_combi_po_angles, dash_angles[..., np.newaxis]), axis=[-2, -3])

    arg_min = np.argmin(all_combi_angle_diffs, axis=-1)

    arg_min_combi = gather_per_rev[arg_min]

    best_pos = allp_pos[arg_min_combi[:, 0], arg_min_combi[:, 1], arg_min_combi[:, 2]]

    ref_po = best_pos

    return (ref_vo, ref_po)

def __generate_samples_per_batch(sb_postar, sb_vo, sb_iseg, counts, sample_size, num_instances):
    samples = []
    for i in range(num_instances):
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

def generate_ref_data(model_info, postar, vo, iseg, counts, sample_size, num_instances=1):
        samples = np.array(
            [
                __generate_samples_per_batch(
                    *args,
                    counts=counts,
                    sample_size=sample_size,
                    num_instances=num_instances
                ) 
                for args in zip(postar, vo, iseg)
            ], dtype=np.float32
        )

        vo_samples = samples[...,0]
        postar_samples = samples[...,1]
    
        if "symmetries_continuous" in model_info:
            print("generate ref samples for continuous symmetries around z")
            #This code does not work
            ref_vo, ref_po = __make_ref_outof_samples_symmetrical(vo_samples, np.array([0, 0, 1], dtype=np.float32))

        elif "symmetries_discrete" in model_info:
            sym_discrete = model_info["symmetries_discrete"]
            if math.isclose(sym_discrete[0][2,2], 1, abs_tol=1e-3):
                factor = len(sym_discrete)+1
                print("generate ref samples discrete symmetries with z_factor=", factor)
                ref_vo, ref_po = __make_ref_outof_samples_symmetrical(
                    vo_samples,
                    get_obj_star0_from_obj_star(postar_samples, z_factor=factor),
                    factor,
                    np.array([0, 0, 1], dtype=np.float32)
                )

            elif math.isclose(sym_discrete[0][1,1], 1, abs_tol=1e-3):
                factor = len(sym_discrete)+1
                print("generate ref samples discrete symmetries with y_factor=", factor)
                ref_vo, ref_po = __make_ref_outof_samples_symmetrical(
                    vo_samples,
                    get_obj_star0_from_obj_star(postar_samples, y_factor=factor),
                    factor,
                    np.array([0, 0, 1], dtype=np.float32)
                )
        else:
            raise ValueError("Something went wrong.")
        return { 'po': ref_po, 'vo': ref_vo }