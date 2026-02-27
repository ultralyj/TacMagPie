# constitutive.py
"""
本构方程模块
当前实现：Yeoh 超弹性模型（三参数）

对外接口
--------
build_yeoh_piola(C1, C2, C3, kappa)
    返回 Taichi @ti.func  yeoh_piola(F_mat)
    计算第一类 Piola-Kirchhoff 应力张量 P
"""

import taichi as ti


def build_yeoh_piola(C1: float, C2: float, C3: float, kappa: float):
    """
    工厂函数：绑定材料参数后返回 Yeoh 本构 ti.func。

    参数
    ----
    C1, C2, C3 : Yeoh 模型系数 (Pa)
    kappa      : 体积惩罚系数（0 = 不可压缩近似）

    返回
    ----
    yeoh_piola : @ti.func，签名 (F_mat: ti.template()) -> ti.Matrix
    """

    @ti.func
    def yeoh_piola(F_mat: ti.template()):
        """
        计算 Yeoh 超弹性模型的第一类 Piola-Kirchhoff 应力。

        参数
        ----
        F_mat : 变形梯度矩阵 (3×3 ti.Matrix)

        返回
        ----
        P : 第一类 Piola-Kirchhoff 应力 (3×3 ti.Matrix)
        """
        J = F_mat.determinant()

        # 防止极度压缩导致数值崩溃
        if J < 0.1:
            F_mat = ti.pow(0.1 / J, 1.0 / 3.0) * F_mat
            J = 0.1

        F_inv_T = F_mat.inverse().transpose()
        I1      = (F_mat.transpose() @ F_mat).trace()

        # 等容第一不变量 Ī₁ = J^(-2/3) · I₁
        J23   = ti.pow(J, -2.0 / 3.0)
        I1b   = J23 * I1
        i1bm3 = I1b - 3.0

        # dW/dĪ₁ = C1 + 2·C2·(Ī₁-3) + 3·C3·(Ī₁-3)²
        dWdI1b = C1 + 2.0 * C2 * i1bm3 + 3.0 * C3 * i1bm3 * i1bm3

        # 等容部分
        P_iso = 2.0 * dWdI1b * J23 * (F_mat - (I1 / 3.0) * F_inv_T)

        # 体积部分（惩罚项）
        P_vol = kappa * (J - 1.0) * J * F_inv_T

        return P_iso + P_vol

    return yeoh_piola
