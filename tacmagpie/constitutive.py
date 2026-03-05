"""Constitutive equation module for hyperelastic materials.

Currently implements the three-parameter Yeoh hyperelastic model for computing
the first Piola-Kirchhoff stress tensor.

Public Interface
----------------
build_yeoh_piola(C1, C2, C3, kappa)
    Returns a Taichi function that computes the first Piola-Kirchhoff stress
    tensor P from the deformation gradient F.
"""

import taichi as ti


def build_yeoh_piola(C1: float, C2: float, C3: float, kappa: float):
    """Build Yeoh constitutive model function with material parameters.

    Args:
        C1: First Yeoh model coefficient (Pa).
        C2: Second Yeoh model coefficient (Pa).
        C3: Third Yeoh model coefficient (Pa).
        kappa: Volumetric penalty coefficient (0 for incompressible approximation).

    Returns:
        Taichi function with signature (F_mat: ti.template()) -> ti.Matrixthat computes the first Piola-Kirchhoff stress tensor.
    """

    @ti.func
    def yeoh_piola(F_mat: ti.template()):
        """Compute first Piola-Kirchhoff stress for Yeoh hyperelastic model.

        Args:
            F_mat: Deformation gradient matrix (3x3 ti.Matrix).

        Returns:
            First Piola-Kirchhoff stress tensor P (3x3 ti.Matrix).
        """
        J = F_mat.determinant()

        if J < 0.1:
            F_mat = ti.pow(0.1 / J, 1.0 / 3.0) * F_mat
            J = 0.1

        F_inv_T = F_mat.inverse().transpose()
        I1 = (F_mat.transpose() @ F_mat).trace()

        J23 = ti.pow(J, -2.0 / 3.0)
        I1b = J23 * I1
        i1bm3 = I1b - 3.0

        dWdI1b = C1 + 2.0 * C2 * i1bm3 + 3.0 * C3 * i1bm3 * i1bm3

        P_iso = 2.0 * dWdI1b * J23 * (F_mat - (I1 / 3.0) * F_inv_T)
        P_vol = kappa * (J - 1.0) * J * F_inv_T

        return P_iso + P_vol

    return yeoh_piola
