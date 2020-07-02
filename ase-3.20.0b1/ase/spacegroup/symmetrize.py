"""
Provides FixSymmetry class to preserve spacegroup symmetry during optimisation
"""
import numpy as np

from ase.constraints import (FixConstraint, voigt_6_to_full_3x3_stress,
                             full_3x3_to_voigt_6_stress)

__all__ = ['refine_symmetry', 'check_symmetry', 'FixSymmetry']


def print_symmetry(symprec, dataset):
    print("ase.spacegroup.symmetrize: prec", symprec,
          "got symmetry group number", dataset["number"],
          ", international (Hermann-Mauguin)", dataset["international"],
          ", Hall ", dataset["hall"])


def refine_symmetry(atoms, symprec=0.01, verbose=False):
    """
    Refine symmetry of an Atoms object

    Parameters
    ----------
    atoms - input Atoms object
    symprec - symmetry precicion
    verbose - if True, print out symmetry information before and after

    Returns
    -------

    spglib dataset

    """
    # check if we have access to get_spacegroup() from spglib
    # https://atztogo.github.io/spglib/
    try:
        import spglib  # For version 1.9 or later
    except ImportError:
        from pyspglib import spglib  # For versions 1.8.x or before

    # test orig config with desired tol
    dataset = check_symmetry(atoms, symprec, verbose=verbose)

    # set actual cell to symmetrized cell vectors by copying
    # transformed and rotated standard cell
    std_cell = dataset['std_lattice']
    trans_std_cell = dataset['transformation_matrix'].T @ std_cell
    rot_trans_std_cell = trans_std_cell @ dataset['std_rotation_matrix']
    atoms.set_cell(rot_trans_std_cell, True)

    # get new dataset and primitive cell
    dataset = check_symmetry(atoms, symprec=symprec, verbose=verbose)
    res = spglib.find_primitive(atoms, symprec=symprec)
    prim_cell, prim_scaled_pos, prim_types = res

    # calculate offset between standard cell and actual cell
    std_cell = dataset['std_lattice']
    rot_std_cell = std_cell @ dataset['std_rotation_matrix']
    rot_std_pos = dataset['std_positions'] @ rot_std_cell
    pos = atoms.get_positions()
    dp0 = (pos[list(dataset['mapping_to_primitive']).index(0)] - rot_std_pos[
        list(dataset['std_mapping_to_primitive']).index(0)])

    # create aligned set of standard cell positions to figure out mapping
    rot_prim_cell = prim_cell @ dataset['std_rotation_matrix']
    inv_rot_prim_cell = np.linalg.inv(rot_prim_cell)
    aligned_std_pos = rot_std_pos + dp0

    # find ideal positions from position of corresponding std cell atom +
    #    integer_vec . primitive cell vectors
    # here we are assuming that primitive vectors returned by find_primitive
    #    are compatible with std_lattice returned by get_symmetry_dataset
    mapping_to_primitive = list(dataset['mapping_to_primitive'])
    std_mapping_to_primitive = list(dataset['std_mapping_to_primitive'])
    pos = atoms.get_positions()
    for i_at in range(len(atoms)):
        std_i_at = std_mapping_to_primitive.index(mapping_to_primitive[i_at])
        dp = aligned_std_pos[std_i_at] - pos[i_at]
        dp_s = dp @ inv_rot_prim_cell
        pos[i_at] = (aligned_std_pos[std_i_at] - np.round(dp_s) @ rot_prim_cell)
    atoms.set_positions(pos)

    # test final config with tight tol
    return check_symmetry(atoms, symprec=1e-4, verbose=verbose)


def check_symmetry(atoms, symprec=1.0e-6, verbose=False):
    """
    Check symmetry of `atoms` with precision `symprec` using `spglib`

    Prints a summary and returns result of `spglib.get_symmetry_dataset()`
    """
    # check if we have access to get_spacegroup from spglib
    # https://atztogo.github.io/spglib/
    try:
        import spglib  # For version 1.9 or later
    except ImportError:
        from pyspglib import spglib  # For versions 1.8.x or before
    dataset = spglib.get_symmetry_dataset(atoms, symprec=symprec)
    if verbose:
        print_symmetry(symprec, dataset)
    return dataset


def is_subgroup(sup_data, sub_data, tol=1e-10):
    """
    Test if spglib dataset `sub_data` is a subgroup of dataset `sup_data`
    """
    for rot1, trns1 in zip(sub_data['rotations'], sub_data['translations']):
        for rot2, trns2 in zip(sup_data['rotations'], sup_data['translations']):
            if np.all(rot1 == rot2) and np.linalg.norm(trns1 - trns2) < tol:
                break
        else:
            return False
    return True


def prep_symmetry(atoms, symprec=1.0e-6, verbose=False):
    """
    Prepare `at` for symmetry-preserving minimisation at precision `symprec`

    Returns a tuple `(rotations, translations, symm_map)`
    """
    # check if we have access to get_spacegroup from spglib
    # https://atztogo.github.io/spglib/
    try:
        import spglib  # For version 1.9 or later
    except ImportError:
        from pyspglib import spglib  # For versions 1.8.x or before

    dataset = spglib.get_symmetry_dataset(atoms, symprec=symprec)
    if verbose:
        print_symmetry(symprec, dataset)
    rotations = dataset['rotations'].copy()
    translations = dataset['translations'].copy()
    symm_map = []
    scaled_pos = atoms.get_scaled_positions()
    for (rot, trans) in zip(rotations, translations):
        this_op_map = [-1] * len(atoms)
        for i_at in range(len(atoms)):
            new_p = rot @ scaled_pos[i_at, :] + trans
            dp = scaled_pos - new_p
            dp -= np.round(dp)
            i_at_map = np.argmin(np.linalg.norm(dp, axis=1))
            this_op_map[i_at] = i_at_map
        symm_map.append(this_op_map)
    return (rotations, translations, symm_map)


def symmetrize_rank1(lattice, inv_lattice, forces, rot, trans, symm_map):
    """
    Return symmetrized forces

    lattice vectors expected as row vectors (same as ASE get_cell() convention),
    inv_lattice is its matrix inverse (get_reciprocal_cell().T)
    """
    scaled_symmetrized_forces_T = np.zeros(forces.T.shape)

    scaled_forces_T = np.dot(inv_lattice.T, forces.T)
    for (r, t, this_op_map) in zip(rot, trans, symm_map):
        transformed_forces_T = np.dot(r, scaled_forces_T)
        scaled_symmetrized_forces_T[:, this_op_map] += transformed_forces_T
    scaled_symmetrized_forces_T /= len(rot)
    symmetrized_forces = (lattice.T @ scaled_symmetrized_forces_T).T

    return symmetrized_forces


def symmetrize_rank2(lattice, lattice_inv, stress_3_3, rot):
    """
    Return symmetrized stress

    lattice vectors expected as row vectors (same as ASE get_cell() convention),
    inv_lattice is its matrix inverse (get_reciprocal_cell().T)
    """
    scaled_stress = np.dot(np.dot(lattice, stress_3_3), lattice.T)

    symmetrized_scaled_stress = np.zeros((3, 3))
    for r in rot:
        symmetrized_scaled_stress += np.dot(np.dot(r.T, scaled_stress), r)
    symmetrized_scaled_stress /= len(rot)

    sym = np.dot(np.dot(lattice_inv, symmetrized_scaled_stress), lattice_inv.T)
    return sym


class FixSymmetry(FixConstraint):
    """
    Constraint to preserve spacegroup symmetry during optimisation.

    Requires spglib package to be available.
    """

    def __init__(self, atoms, symprec=0.01, adjust_positions=True,
                 adjust_cell=True, verbose=False):
        self.verbose = verbose
        refine_symmetry(atoms, symprec, self.verbose)  # refine initial symmetry
        sym = prep_symmetry(atoms, symprec, self.verbose)
        self.rotations, self.translations, self.symm_map = sym
        self.do_adjust_positions = adjust_positions
        self.do_adjust_cell = adjust_cell

    def adjust_cell(self, atoms, cell):
        if not self.do_adjust_cell:
            return
        # stress should definitely be symmetrized as a rank 2 tensor
        # UnitCellFilter uses deformation gradient as cell DOF with steps
        # dF = stress.F^-T quantity that should be symmetrized is therefore dF .
        # F^T assume prev F = I, so just symmetrize dF
        cur_cell = atoms.get_cell()
        cur_cell_inv = atoms.get_reciprocal_cell().T

        # F defined such that cell = cur_cell . F^T
        # assume prev F = I, so dF = F - I
        delta_deform_grad = np.dot(cur_cell_inv, cell).T - np.eye(3)
        symmetrized_delta_deform_grad = symmetrize_rank2(cur_cell, cur_cell_inv,
                                                         delta_deform_grad,
                                                         self.rotations)
        cell[:] = np.dot(cur_cell,
                         (symmetrized_delta_deform_grad + np.eye(3)).T)

    def adjust_positions(self, atoms, new):
        if not self.do_adjust_positions:
            return
        # symmetrize changes in position as rank 1 tensors
        step = new - atoms.positions
        symmetrized_step = symmetrize_rank1(atoms.get_cell(),
                                            atoms.get_reciprocal_cell().T, step,
                                            self.rotations, self.translations,
                                            self.symm_map)
        new[:] = atoms.positions + symmetrized_step

    def adjust_forces(self, atoms, forces):
        # symmetrize forces as rank 1 tensors
        # print('adjusting forces')
        forces[:] = symmetrize_rank1(atoms.get_cell(),
                                     atoms.get_reciprocal_cell().T, forces,
                                     self.rotations, self.translations,
                                     self.symm_map)

    def adjust_stress(self, atoms, stress):
        # symmetrize stress as rank 2 tensor
        raw_stress = voigt_6_to_full_3x3_stress(stress)
        symmetrized_stress = symmetrize_rank2(atoms.get_cell(),
                                              atoms.get_reciprocal_cell().T,
                                              raw_stress, self.rotations)
        stress[:] = full_3x3_to_voigt_6_stress(symmetrized_stress)

    def index_shuffle(self, atoms, ind):
        if len(atoms) != len(ind) or len(set(ind)) != len(ind):
            raise RuntimeError("FixSymmetry can only accomodate atom"
                               " permutions, and len(Atoms) == {} "
                               "!= len(ind) == {} or ind has duplicates"
                               .format(len(atoms), len(ind)))

        ind_reversed = np.zeros((len(ind)), dtype=int)
        ind_reversed[ind] = range(len(ind))
        new_symm_map = []
        for sm in self.symm_map:
            new_sm = np.array([-1] * len(atoms))
            for at_i in range(len(ind)):
                new_sm[ind_reversed[at_i]] = ind_reversed[sm[at_i]]
            new_symm_map.append(new_sm)

        self.symm_map = new_symm_map
