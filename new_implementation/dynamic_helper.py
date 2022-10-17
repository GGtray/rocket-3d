from numpy  import tan, arctan, sqrt, pi, sin, cos, matmul
import math

def M1_gamma(gamma, x):
    a = gamma
    M1 = [
            [1,        0,       0],
            [0,  cos(a), sin(a)],
            [0, -sin(a), cos(a)]
        ]
    return matmul(M1, x)


def M2_psi(psi, x):
    a = psi
    M2 = [
            [cos(a), 0,-sin(a)],
            [     0, 1,      0],
            [sin(a), 0, cos(a)]
        ]
    return matmul(M2, x)



def M3_phi(phi, x):
    a = phi
    M3 = [
            [ cos(a),  sin(a), 0],
            [-sin(a),  cos(a), 0],
            [      0,       0, 1]
        ]
    return matmul(M3, x)


def thrust_convert_phi_psi_to_1_2(theta_phi, theta_psi):
    """
    目的：将推力的phi，psi描述变换为theta1，theta2描述
    在箭体系下 
    theta_phi: x轴正向绕z轴正向按右手定则转theta_phi角
    theta_psi: x轴正向绕y轴正向按右手定则转theta_psi角

    theta_1: 
    theta_2: y轴绕x轴正向，按右手定则确定正负，范围为（-pi，pi）
    """
    # 简化表示
    phi = theta_phi
    psi = theta_psi

    theta_1 = arctan(sqrt(pow(tan(phi), 2)+ pow(tan(psi), 2)))



    if (phi > 0 and psi > 0) or (phi > 0 and psi < 0):
        theta_2 = -arctan(tan(psi)/tan(phi))
    elif phi < 0 and psi > 0:
        theta_2 = -pi - arctan(tan(psi)/tan(phi))
    elif phi < 0 and psi < 0:
        theta_2 = pi - arctan(tan(psi)/tan(phi))
    else: # zero or pi
        if psi == 0:
            theta_2 = 0
        elif psi < 0 and phi == 0:
            theta_2 = pi / 2
        elif psi > 0 and phi == 0:
            theta_2 = -pi / 2
    return theta_1, theta_2


def thrust_to_fx_fy_fz(F, theta_1, theta_2):
    fx = F * cos(theta_1)
    fy = F * sin(theta_1) * cos(theta_2)
    fz = F * sin(theta_1) * sin(theta_2)
    return fx, fy, fz


def omage_b_to_valtitude(ox, oy, oz, phi, psi, gamma):
    vphi = (oy * sin(gamma) + oz * cos(gamma)) / cos(psi)
    vpsi = oy * cos(gamma) - oz * sin(gamma)
    vgamma = ox + tan(psi) * (oy * sin(gamma) + oz * cos(gamma))

    return vphi, vpsi, vgamma

def test_thrust_convert_phi_psi_to_1_2():
    print(pi/4, -pi + pi/4)
    print(thrust_convert_phi_psi_to_1_2(pi/6, pi/6))
    print(thrust_convert_phi_psi_to_1_2(-pi/6, pi/6))
    print(thrust_convert_phi_psi_to_1_2(pi/6, -pi/6))
    print(thrust_convert_phi_psi_to_1_2(-pi/6, -pi/6))

def test_thrust_to_fx_fy_fz():
    print(thrust_to_fx_fy_fz(10, pi/4, pi/4))

if __name__ == "__main__":
    # test_thrust_to_fx_fy_fz()
    # print(M1_gamma(-pi/5, [3, 1, 2]))
    # print(M1_gamma(pi/5, [3, 1, 2]))
    # print(M1_gamma(-pi/5, M1_gamma(pi/5, [3, 1, 2])))
    ag = pi/4
    x = [1, 2, 3]
    Mx = M1_gamma(ag, M2_psi(ag, M3_phi(ag, x)))
    print(Mx)
    print(-ag)
    x = M3_phi(-ag, M2_psi(-ag, M1_gamma(-ag, Mx)))
    print(x)
    



