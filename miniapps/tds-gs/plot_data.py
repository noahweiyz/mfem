import numpy as np
import matplotlib.pyplot as plt



rr = [8.1953, 8.0494, 7.7160, 7.2383, 6.6367, 5.8737, 4.9252, 4.4863, 4.2837, 4.1938, 4.2156, 4.3369, 4.5249, 4.7482, 5.0601, 5.6576, 6.3491, 6.9774, 7.5117, 7.9219, 8.1621]
zz = [0.6429, 1.5703, 2.3438, 2.9963, 3.5269, 3.9375, 3.9141, 3.2344, 2.3672, 1.3828, 0.3516, -0.6094, -1.5000, -2.3672, -3.3323, -3.0234, -2.5547, -2.0391, -1.4409, -0.7284, 0.1172]

rz = np.array([[8.195284, 0.642945],
               [8.056269, 1.546875],
               [7.730469, 2.318396],
               [7.277213, 2.953125],
               [6.691406, 3.487306],
               [5.980469, 3.895953],
               [5.023438, 3.970442],
               [4.531250, 3.355147],
               [4.310886, 2.531250],
               [4.203125, 1.587245],
               [4.201888, 0.562500],
               [4.298682, -0.375000],
               [4.469732, -1.265625],
               [4.680117, -2.109375],
               [4.913674, -2.953125],
               [5.415017, -3.164062],
               [6.117188, -2.719791],
               [6.746094, -2.244396],
               [7.315237, -1.687500],
               [7.760603, -1.054688],
               [8.078908, -0.281250]])

plt.figure()
plt.plot(rr, zz, '*b')
plt.plot(rz[:, 0], rz[:, 1], 'or')
plt.show()
asdf

def _VacuumVesselMetalWallCoordinates(return_coarse=3):
    """Return r and z coordinates for plasma-facing metal wall"""
    r = np.array([6.2670, 7.2830, 7.8990, 8.3060, 8.3950, 8.2700, 7.9040,
                  7.4000, 6.5870, 5.7530, 4.9040, 4.3110, 4.1260, 4.0760,
                  4.0460, 4.0460, 4.0670, 4.0970, 4.1780, 3.9579, 4.0034,
                  4.1742, 4.3257, 4.4408, 4.5066, 4.5157, 4.4670, 4.4064,
                  4.4062, 4.3773, 4.3115, 4.2457, 4.1799, 4.4918, 4.5687,
                  4.6456, 4.8215, 4.9982, 5.1496, 5.2529, 5.2628, 5.2727,
                  5.5650, 5.5650, 5.5650, 5.5650, 5.5720, 5.5720, 5.5720,
                  5.5720, 5.6008, 5.6842, 5.8150, 5.9821, 6.1710, 6.3655])

    z = np.array([-3.0460, -2.2570, -1.3420, -0.4210,  0.6330,  1.6810,
                   2.4640,  3.1790,  3.8940,  4.5320,  4.7120,  4.3240,
                   3.5820,  2.5660,  1.5490,  0.5330, -0.4840, -1.5000,
                  -2.5060, -2.5384, -2.5384, -2.5674, -2.6514, -2.7808,
                  -2.9410, -3.1139, -3.2801, -3.4043, -3.4048, -3.4799,
                  -3.6148, -3.7497, -3.8847, -3.9092, -3.8276, -3.7460,
                  -3.7090, -3.7414, -3.8382, -3.9852, -4.1244, -4.2636,
                  -4.5559, -4.4026, -4.2494, -4.0962, -3.9961, -3.9956,
                  -3.8960, -3.8950, -3.7024, -3.5265, -3.3823, -3.2822,
                  -3.2350, -3.2446])

    if return_coarse in (2, 3):
        z[18] += 0.2
    if return_coarse == 2:
        points_to_delete = [20, 28, 44, 46, 48]
        r = np.delete(r, points_to_delete, 0)
        z = np.delete(z, points_to_delete, 0)
    if return_coarse == 3:
        points_to_delete = [20, 21, 23, 25, 27, 28, 29, 30, 31, 34, 36, 38,
                            40, 43, 44, 45, 46, 47, 49, 50, 52, 54]
        r = np.delete(r, points_to_delete, 0)
        z = np.delete(z, points_to_delete, 0)
    return r, z


def _VacuumVesselSecondWallCoordinates(return_coarse=4):
    """Return r and z coordinates for outer vacuum vessel wall."""
    r = np.array([6.2270, 6.4090, 6.5880, 6.7639, 6.9365, 7.1054, 7.2706,
                  7.4318, 7.5888, 7.7414, 7.8895, 8.0328, 8.1712, 8.3046,
                  8.4327, 8.5554, 8.6726, 8.7840, 8.8897, 8.9894, 9.0830,
                  9.1705, 9.2516, 9.3264, 9.3946, 9.4563, 9.5114, 9.5598,
                  9.6014, 9.6361, 9.6616, 9.6771, 9.6825, 9.6777, 9.6629,
                  9.6380, 9.6031, 9.5583, 9.5038, 9.4396, 9.3660, 9.2900,
                  9.2141, 9.1382, 9.0623, 8.9863, 8.9104, 8.8345, 8.7585,
                  8.6826, 8.6067, 8.5308, 8.4548, 8.3789, 8.3030, 8.2270,
                  8.1511, 8.0681, 7.9748, 7.8714, 7.7584, 7.6362, 7.5052,
                  7.3659, 7.2188, 7.0645, 6.9035, 6.7364, 6.5638, 6.3864,
                  6.2048, 6.0196, 5.8317, 5.6415, 5.4499, 5.2575, 5.0651,
                  4.8734, 4.6848, 4.5011, 4.3243, 4.1561, 3.9982, 3.8523,
                  3.7199, 3.6022, 3.5005, 3.4159, 3.3492, 3.3010, 3.2719,
                  3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
                  3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
                  3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
                  3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
                  3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
                  3.2622, 3.2622, 3.2622, 3.2622, 3.2717, 3.3002, 3.3473,
                  3.4126, 3.4953, 3.5948, 3.7098, 3.8394, 3.9820, 4.1364,
                  4.3008, 4.4737, 4.6532, 4.8274, 5.0040, 5.1820, 5.3605,
                  5.5385, 5.7151, 5.8894, 6.0603, 6.2270])

    z = np.array([ 5.4684,  5.3873,  5.3000,  5.2064,  5.1067,  5.0011,
                   4.8897,  4.7726,  4.6499,  4.5218,  4.3885,  4.2501,
                   4.1068,  3.9588,  3.8062,  3.6492,  3.4881,  3.3229,
                   3.1540,  2.9815,  2.8056,  2.6266,  2.4447,  2.2600,
                   2.0728,  1.8833,  1.6919,  1.4986,  1.3037,  1.1076,
                   0.9153,  0.7220,  0.5282,  0.3343,  0.1410, -0.0513,
                  -0.2421, -0.4308, -0.6169, -0.7999, -0.9793, -1.1514,
                  -1.3235, -1.4956, -1.6677, -1.8398, -2.0119, -2.1840,
                  -2.3561, -2.5282, -2.7003, -2.8724, -3.0445, -3.2166,
                  -3.3887, -3.5608, -3.7329, -3.9066, -4.0749, -4.2373,
                  -4.3931, -4.5418, -4.6828, -4.8157, -4.9398, -5.0549,
                  -5.1604, -5.2559, -5.3412, -5.4158, -5.4796, -5.5323,
                  -5.5737, -5.6036, -5.6220, -5.6287, -5.6238, -5.6034,
                  -5.5637, -5.5052, -5.4285, -5.3343, -5.2237, -5.0977,
                  -4.9576, -4.8049, -4.6411, -4.4679, -4.2871, -4.1004,
                  -3.9099, -3.7173, -3.5200, -3.3226, -3.1253, -2.9280,
                  -2.7306, -2.5333, -2.3359, -2.1386, -1.9412, -1.7439,
                  -1.5466, -1.3492, -1.1519, -0.9545, -0.7572, -0.5599,
                  -0.3625, -0.1652,  0.0322,  0.2295,  0.4269,  0.6242,
                   0.8215,  1.0189,  1.2162,  1.4136,  1.6109,  1.8083,
                   2.0056,  2.2029,  2.4003,  2.5976,  2.7950,  2.9923,
                   3.1897,  3.3870,  3.5843,  3.7817,  3.9697,  4.1559,
                   4.3382,  4.5148,  4.6839,  4.8438,  4.9929,  5.1295,
                   5.2524,  5.3603,  5.4520,  5.5266,  5.5834,  5.6223,
                   5.6483,  5.6614,  5.6614,  5.6484,  5.6224,  5.5836,
                   5.5322,  5.4684])

    return r[:-1:return_coarse], z[:-1:return_coarse]

def _VacuumVesselFirstWallCoordinates(return_coarse=5):
    """Return r and z coordinates for inner vacuum vessel wall"""
    r = np.array([3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5396,
                  3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5396,
                  3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5514, 3.5869,
                  3.6453, 3.7260, 3.8276, 3.9489, 4.0878, 4.2425, 4.4107,
                  4.5899, 4.7775, 4.9708, 5.1670, 5.3631, 5.5564, 5.7440,
                  5.9232, 6.0913, 6.2474, 6.4034, 6.5595, 6.7156, 6.8716,
                  7.0315, 7.1867, 7.3371, 7.4825, 7.6226, 7.7572, 7.8861,
                  8.0092, 8.1262, 8.2370, 8.3413, 8.4391, 8.5302, 8.6145,
                  8.6918, 8.7619, 8.8249, 8.8764, 8.9166, 8.9453, 8.9638,
                  8.9768, 8.9844, 8.9865, 8.9832, 8.9744, 8.9602, 8.9364,
                  8.8985, 8.8468, 8.7817, 8.7023, 8.6228, 8.5433, 8.4638,
                  8.3844, 8.3049, 8.2254, 8.1459, 8.0665, 7.9870, 7.9075,
                  7.8280, 7.7486, 7.6691, 7.5896, 7.5102, 7.4307, 7.3458,
                  7.2486, 7.1396, 7.0195, 6.8888, 6.7483, 6.5988, 6.4411,
                  6.2762, 6.1048, 5.9281, 5.7469, 5.5623, 5.3753, 5.1869,
                  4.9897, 4.7949, 4.6054, 4.4242, 4.2538, 4.0969, 3.9558,
                  3.8327, 3.7293, 3.6472, 3.5877, 3.5516, 3.5396, 3.5396,
                  3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5396,
                  3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5396, 3.5396,
                  3.5396, 3.5396])

    z = np.array([ 0.0307,  0.2278,  0.4250,  0.6221,  0.8192,  1.0164,
                   1.2135,  1.4106,  1.6078,  1.8049,  2.0020,  2.1992,
                   2.3963,  2.5935,  2.7906,  2.9877,  3.1849,  3.3820,
                   3.5791,  3.7753,  3.9686,  4.1562,  4.3354,  4.5035,
                   4.6582,  4.7971,  4.9183,  5.0200,  5.1006,  5.1590,
                   5.1944,  5.2062,  5.1943,  5.1589,  5.1004,  5.0197,
                   4.9180,  4.8102,  4.7024,  4.5946,  4.4868,  4.3790,
                   4.2640,  4.1429,  4.0158,  3.8830,  3.7446,  3.6009,
                   3.4521,  3.2984,  3.1400,  2.9772,  2.8102,  2.6393,
                   2.4647,  2.2867,  2.1056,  1.9216,  1.7351,  1.5529,
                   1.3679,  1.1809,  1.0073,  0.8333,  0.6590,  0.4845,
                   0.3100,  0.1357, -0.0382, -0.2088, -0.3768, -0.5411,
                  -0.7006, -0.8743, -1.0480, -1.2217, -1.3954, -1.5691,
                  -1.7428, -1.9166, -2.0903, -2.2640, -2.4377, -2.6114,
                  -2.7851, -2.9588, -3.1325, -3.3062, -3.4799, -3.6536,
                  -3.8222, -3.9840, -4.1380, -4.2835, -4.4196, -4.5456,
                  -4.6608, -4.7645, -4.8561, -4.9352, -5.0013, -5.0540,
                  -5.0931, -5.1183, -5.1295, -5.1217, -5.0899, -5.0345,
                  -4.9564, -4.8567, -4.7369, -4.5989, -4.4446, -4.2765,
                  -4.0970, -3.9088, -3.7148, -3.5178, -3.3206, -3.1235,
                  -2.9264, -2.7292, -2.5321, -2.3349, -2.1378, -1.9407,
                  -1.7435, -1.5464, -1.3493, -1.1521, -0.9550, -0.7579,
                  -0.5607, -0.3636, -0.1665])

    return r[::return_coarse], z[::return_coarse]


plt.figure()
plt.plot(*_VacuumVesselMetalWallCoordinates(), "r.-")
plt.plot(*_VacuumVesselSecondWallCoordinates(4), "b.-")
plt.plot(*_VacuumVesselFirstWallCoordinates(5), "g.-")

plt.show()

asdf
rz = [6.267000e+00, -3.046000e+00, 7.283000e+00, -2.257000e+00, 7.899000e+00, -1.342000e+00, 8.306000e+00, -4.210000e-01, 8.395000e+00, 6.330000e-01, 8.270000e+00, 1.681000e+00, 7.904000e+00, 2.464000e+00, 7.400000e+00, 3.179000e+00, 6.587000e+00, 3.894000e+00, 5.753000e+00, 4.532000e+00, 4.904000e+00, 4.712000e+00, 4.311000e+00, 4.324000e+00, 4.126000e+00, 3.582000e+00, 4.076000e+00, 2.566000e+00, 4.046000e+00, 1.549000e+00, 4.046000e+00, 5.330000e-01, 4.067000e+00, -4.840000e-01, 4.097000e+00, -1.500000e+00, 4.178000e+00, -2.506000e+00, 3.957900e+00, -2.538400e+00, 4.003400e+00, -2.538400e+00, 4.174200e+00, -2.567400e+00, 4.325700e+00, -2.651400e+00, 4.440800e+00, -2.780800e+00, 4.506600e+00, -2.941000e+00, 4.515700e+00, -3.113900e+00, 4.467000e+00, -3.280100e+00, 4.406400e+00, -3.404300e+00, 4.406200e+00, -3.404800e+00, 4.377300e+00, -3.479900e+00, 4.311500e+00, -3.614800e+00, 4.245700e+00, -3.749700e+00, 4.179900e+00, -3.884700e+00, 4.491800e+00, -3.909200e+00, 4.568700e+00, -3.827600e+00, 4.645600e+00, -3.746000e+00, 4.821500e+00, -3.709000e+00, 4.998200e+00, -3.741400e+00, 5.149600e+00, -3.838200e+00, 5.252900e+00, -3.985200e+00, 5.262800e+00, -4.124400e+00, 5.272700e+00, -4.263600e+00, 5.565000e+00, -4.555900e+00, 5.565000e+00, -4.402600e+00, 5.565000e+00, -4.249400e+00, 5.565000e+00, -4.096200e+00, 5.572000e+00, -3.996100e+00, 5.572000e+00, -3.995600e+00, 5.572000e+00, -3.896000e+00, 5.572000e+00, -3.895000e+00, 5.600800e+00, -3.702400e+00, 5.684200e+00, -3.526500e+00, 5.815000e+00, -3.382300e+00, 5.982100e+00, -3.282200e+00, 6.171000e+00, -3.235000e+00, 6.365500e+00, -3.244600e+00]

r = rz[0::2]
z = rz[1::2]

curve1 = (r[0:19], z[0:19])
curve2 = (r[19:33], z[19:33])
curve3 = (r[33:42], z[33:42])
curve4 = (r[42:], z[42:])



plt.plot(r, z, '.-')
plt.show()

i = 0
for r_, z_ in zip(r, z):
    plt.text(r_, z_, i)
    i += 1

plt.show()

breakpoint()


r = np.array([6.2270, 6.4090, 6.5880, 6.7639, 6.9365, 7.1054, 7.2706,
              7.4318, 7.5888, 7.7414, 7.8895, 8.0328, 8.1712, 8.3046,
              8.4327, 8.5554, 8.6726, 8.7840, 8.8897, 8.9894, 9.0830,
              9.1705, 9.2516, 9.3264, 9.3946, 9.4563, 9.5114, 9.5598,
              9.6014, 9.6361, 9.6616, 9.6771, 9.6825, 9.6777, 9.6629,
              9.6380, 9.6031, 9.5583, 9.5038, 9.4396, 9.3660, 9.2900,
              9.2141, 9.1382, 9.0623, 8.9863, 8.9104, 8.8345, 8.7585,
              8.6826, 8.6067, 8.5308, 8.4548, 8.3789, 8.3030, 8.2270,
              8.1511, 8.0681, 7.9748, 7.8714, 7.7584, 7.6362, 7.5052,
              7.3659, 7.2188, 7.0645, 6.9035, 6.7364, 6.5638, 6.3864,
              6.2048, 6.0196, 5.8317, 5.6415, 5.4499, 5.2575, 5.0651,
              4.8734, 4.6848, 4.5011, 4.3243, 4.1561, 3.9982, 3.8523,
              3.7199, 3.6022, 3.5005, 3.4159, 3.3492, 3.3010, 3.2719,
              3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
              3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
              3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
              3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
              3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622, 3.2622,
              3.2622, 3.2622, 3.2622, 3.2622, 3.2717, 3.3002, 3.3473,
              3.4126, 3.4953, 3.5948, 3.7098, 3.8394, 3.9820, 4.1364,
              4.3008, 4.4737, 4.6532, 4.8274, 5.0040, 5.1820, 5.3605,
              5.5385, 5.7151, 5.8894, 6.0603, 6.2270])

z = np.array([ 5.4684,  5.3873,  5.3000,  5.2064,  5.1067,  5.0011,
               4.8897,  4.7726,  4.6499,  4.5218,  4.3885,  4.2501,
               4.1068,  3.9588,  3.8062,  3.6492,  3.4881,  3.3229,
               3.1540,  2.9815,  2.8056,  2.6266,  2.4447,  2.2600,
               2.0728,  1.8833,  1.6919,  1.4986,  1.3037,  1.1076,
               0.9153,  0.7220,  0.5282,  0.3343,  0.1410, -0.0513,
              -0.2421, -0.4308, -0.6169, -0.7999, -0.9793, -1.1514,
              -1.3235, -1.4956, -1.6677, -1.8398, -2.0119, -2.1840,
              -2.3561, -2.5282, -2.7003, -2.8724, -3.0445, -3.2166,
              -3.3887, -3.5608, -3.7329, -3.9066, -4.0749, -4.2373,
              -4.3931, -4.5418, -4.6828, -4.8157, -4.9398, -5.0549,
              -5.1604, -5.2559, -5.3412, -5.4158, -5.4796, -5.5323,
              -5.5737, -5.6036, -5.6220, -5.6287, -5.6238, -5.6034,
              -5.5637, -5.5052, -5.4285, -5.3343, -5.2237, -5.0977,
              -4.9576, -4.8049, -4.6411, -4.4679, -4.2871, -4.1004,
              -3.9099, -3.7173, -3.5200, -3.3226, -3.1253, -2.9280,
              -2.7306, -2.5333, -2.3359, -2.1386, -1.9412, -1.7439,
              -1.5466, -1.3492, -1.1519, -0.9545, -0.7572, -0.5599,
              -0.3625, -0.1652,  0.0322,  0.2295,  0.4269,  0.6242,
               0.8215,  1.0189,  1.2162,  1.4136,  1.6109,  1.8083,
               2.0056,  2.2029,  2.4003,  2.5976,  2.7950,  2.9923,
               3.1897,  3.3870,  3.5843,  3.7817,  3.9697,  4.1559,
               4.3382,  4.5148,  4.6839,  4.8438,  4.9929,  5.1295,
               5.2524,  5.3603,  5.4520,  5.5266,  5.5834,  5.6223,
               5.6483,  5.6614,  5.6614,  5.6484,  5.6224,  5.5836,
               5.5322,  5.4684])

breakpoint()
fbar = [1, 0.99199, 0.983911, 0.975783, 0.967616, 0.959415, 0.951183, 0.942926, 0.934644, 0.926339, 0.918014, 0.909669, 0.901305, 0.892923, 0.884525, 0.876111, 0.867681, 0.859235, 0.850776, 0.842302, 0.833815, 0.825314, 0.816801, 0.808275, 0.799737, 0.791187, 0.782625, 0.774051, 0.765466, 0.756869, 0.748262, 0.739644, 0.731016, 0.722377, 0.713728, 0.705069, 0.696399, 0.687719, 0.679028, 0.670328, 0.661619, 0.652901, 0.644177, 0.635445, 0.626704, 0.617952, 0.60919, 0.600417, 0.591629, 0.582828, 0.574013, 0.565192, 0.556369, 0.547551, 0.538747, 0.52997, 0.521236, 0.512558, 0.503949, 0.495419, 0.486975, 0.478626, 0.47038, 0.462238, 0.454199, 0.446266, 0.438436, 0.43071, 0.423082, 0.415553, 0.408117, 0.400773, 0.393521, 0.38636, 0.37929, 0.372312, 0.365424, 0.358626, 0.351918, 0.345298, 0.338767, 0.332322, 0.325965, 0.319692, 0.313504, 0.307401, 0.30138, 0.295441, 0.289584, 0.283808, 0.278111, 0.272494, 0.266955, 0.261494, 0.25611, 0.250802, 0.245571, 0.240413, 0.235331, 0.230321, 0.225383, 0.220517, 0.215722, 0.210998, 0.206343, 0.201757, 0.197238, 0.192787, 0.188403, 0.184085, 0.179832, 0.175643, 0.171518, 0.167457, 0.163457, 0.159519, 0.155642, 0.151825, 0.148068, 0.14437, 0.14073, 0.137147, 0.133622, 0.130153, 0.126739, 0.12338, 0.120075, 0.116824, 0.113626, 0.11048, 0.107386, 0.104343, 0.101351, 0.0984078, 0.0955144, 0.092669, 0.0898723, 0.0871224, 0.0844198, 0.0817633, 0.0791522, 0.0765866, 0.074065, 0.0715876, 0.0691536, 0.0667617, 0.0644126, 0.0621056, 0.0598387, 0.0576134, 0.0554275, 0.0532811, 0.0511742, 0.0491055, 0.0470743, 0.0450813, 0.0431251, 0.0412052, 0.0393214, 0.0374725, 0.0356592, 0.0338801, 0.0321352, 0.0304239, 0.0287455, 0.0270993, 0.0254861, 0.0239038, 0.0223531, 0.0208333, 0.0193438, 0.0178846, 0.0164543, 0.015053, 0.0136813, 0.0123366, 0.0110209, 0.00973211, 0.00847035, 0.00723557, 0.00602712, 0.00484434, 0.00368657, 0.00255447, 0.00144738, 0.000363983, -0.000694399, -0.00172974, -0.00274205, -0.00373067, -0.00469756, -0.00564141, -0.00656355, -0.00746462, -0.00834397, -0.00920292, -0.0100402, -0.0108583, -0.011656, -0.012434, -0.0131923, -0.0139321, -0.0146521, -0.0153544, -0.0160376, -0.0167031, -0.0173508, -0.0179806, -0.0185928, -0.0191878, -0.0197657, -0.0203265, -0.0208708, -0.021398, -0.0219094, -0.0224044, -0.0228836, -0.0233463, -0.0237939, -0.0242263, -0.0246429, -0.0250444, -0.0254308, -0.0258027, -0.0261594, -0.0265023, -0.0268301, -0.0271441, -0.0274436, -0.0277286, -0.0279991, -0.0282558, -0.028498, -0.0287257, -0.0289383, -0.0291364, -0.0293181, -0.0294813, -0.0296235, -0.0297387, -0.029817, -0.0298433, -0.0297881, -0.029611, -0.0292266, -0.0285342, -0.027408, -0.025725, -0.023468, -0.0206997, -0.0175923, -0.0143592, -0.011215, -0.00839268, -0.00593036, -0.0033397, -0]
fbar_prime = [-2.04187, -2.05939, -2.07447, -2.08576, -2.09511, -2.10337, -2.11061, -2.1171, -2.12308, -2.12864, -2.13378, -2.13875, -2.14347, -2.14777, -2.15198, -2.15611, -2.16007, -2.16378, -2.1674, -2.17102, -2.17448, -2.17776, -2.18105, -2.18425, -2.18728, -2.19031, -2.19343, -2.19638, -2.19924, -2.20211, -2.2048, -2.2075, -2.21011, -2.21281, -2.2155, -2.21812, -2.22081, -2.22351, -2.22604, -2.22839, -2.23059, -2.23252, -2.23446, -2.23657, -2.23901, -2.24171, -2.24457, -2.24786, -2.25139, -2.25485, -2.25738, -2.25847, -2.25805, -2.25561, -2.2503, -2.24137, -2.22882, -2.21272, -2.19377, -2.17271, -2.14945, -2.12418, -2.09772, -2.0711, -2.04439, -2.01769, -1.99123, -1.96528, -1.94009, -1.91558, -1.89173, -1.86831, -1.84489, -1.82147, -1.79822, -1.77488, -1.75171, -1.7288, -1.70605, -1.68339, -1.66089, -1.63865, -1.61666, -1.59492, -1.57327, -1.55187, -1.53081, -1.50992, -1.48911, -1.46855, -1.44816, -1.42794, -1.40797, -1.38818, -1.36855, -1.349, -1.32979, -1.31075, -1.29188, -1.27326, -1.25481, -1.23661, -1.2185, -1.20055, -1.18286, -1.16542, -1.14807, -1.13088, -1.11394, -1.09709, -1.0805, -1.06415, -1.04789, -1.03189, -1.01605, -1.00029, -0.98479, -0.969457, -0.954292, -0.939296, -0.924468, -0.909808, -0.895318, -0.881079, -0.866926, -0.85294, -0.839123, -0.825475, -0.811995, -0.798684, -0.785625, -0.772566, -0.759676, -0.747039, -0.73457, -0.722185, -0.709969, -0.697921, -0.685958, -0.674247, -0.662621, -0.651163, -0.639874, -0.628668, -0.617716, -0.606848, -0.59598, -0.585449, -0.575002, -0.564639, -0.554529, -0.544419, -0.534478, -0.524789, -0.5151, -0.505496, -0.496144, -0.486877, -0.477778, -0.468763, -0.459833, -0.451071, -0.442393, -0.433884, -0.425543, -0.417203, -0.40903, -0.401027, -0.393023, -0.385188, -0.377437, -0.369855, -0.362441, -0.354942, -0.347697, -0.340536, -0.333375, -0.326466, -0.319558, -0.312733, -0.306078, -0.299591, -0.293103, -0.286616, -0.280382, -0.274147, -0.267997, -0.2621, -0.256118, -0.250305, -0.244576, -0.238847, -0.233371, -0.227894, -0.222502, -0.217111, -0.211887, -0.206832, -0.201693, -0.196638, -0.191751, -0.186865, -0.182063, -0.177345, -0.172627, -0.168077, -0.163528, -0.158978, -0.154513, -0.150132, -0.145751, -0.141455, -0.137158, -0.132945, -0.128817, -0.124689, -0.120561, -0.116517, -0.112641, -0.108682, -0.104722, -0.100846, -0.0970552, -0.093264, -0.089557, -0.0858501, -0.0821431, -0.0785204, -0.0748134, -0.0711064, -0.0674837, -0.063861, -0.060154, -0.0563628, -0.0525716, -0.0486119, -0.0441466, -0.0390917, -0.0329415, -0.0247693, -0.0133956, 0.00370697, 0.02974, 0.0718647, 0.137832, 0.232781, 0.359576, 0.504316, 0.643244, 0.752094, 0.811574, 0.816292, 0.76372, 0.676438, 0.646782, 0.759086, 0.950838]
fbar_double_prime = [-0.00775094, -4.48611, -3.23517, -2.545, -2.24305, -1.98424, -1.72543, -1.59602, -1.46661, -1.38034, -1.25093, -1.29407, -1.12153, -1.07839, -1.07839, -1.03526, -0.99212, -0.905849, -0.948984, -0.905849, -0.862713, -0.819577, -0.862713, -0.776442, -0.776442, -0.776442, -0.819577, -0.69017, -0.776442, -0.69017, -0.69017, -0.69017, -0.647035, -0.733306, -0.647035, -0.69017, -0.69017, -0.69017, -0.603899, -0.603899, -0.517628, -0.474492, -0.517628, -0.560764, -0.69017, -0.69017, -0.776442, -0.905849, -0.905849, -0.862713, -0.431357, -0.129407, 0.345085, 0.905849, 1.8117, 2.76068, 3.66653, 4.57238, 5.13314, 5.65077, 6.25467, 6.68603, 6.85857, 6.7723, 6.9017, 6.7723, 6.7723, 6.51348, 6.38408, 6.1684, 6.03899, 5.95272, 6.03899, 5.95272, 5.95272, 5.99586, 5.86645, 5.86645, 5.78018, 5.82331, 5.69391, 5.69391, 5.5645, 5.5645, 5.52136, 5.43509, 5.34882, 5.34882, 5.30569, 5.21941, 5.21941, 5.13314, 5.09001, 5.04687, 5.00374, 5.00374, 4.83119, 4.91746, 4.74492, 4.78806, 4.65865, 4.65865, 4.61552, 4.57238, 4.48611, 4.44297, 4.44297, 4.3567, 4.31357, 4.31357, 4.18416, 4.18416, 4.14102, 4.05475, 4.05475, 4.01162, 3.92534, 3.92534, 3.83907, 3.83907, 3.7528, 3.7528, 3.66653, 3.62339, 3.62339, 3.53712, 3.53712, 3.45085, 3.45085, 3.36458, 3.32145, 3.36458, 3.23517, 3.23517, 3.1489, 3.19204, 3.06263, 3.10577, 3.0195, 2.97636, 2.97636, 2.89009, 2.89009, 2.84695, 2.76068, 2.80382, 2.76068, 2.63127, 2.71755, 2.58814, 2.58814, 2.58814, 2.50187, 2.45873, 2.50187, 2.4156, 2.37246, 2.37246, 2.28619, 2.32933, 2.24305, 2.24305, 2.19992, 2.15678, 2.11365, 2.15678, 2.02738, 2.07051, 2.02738, 1.98424, 1.98424, 1.89797, 1.89797, 1.9411, 1.76856, 1.89797, 1.76856, 1.76856, 1.76856, 1.72543, 1.68229, 1.63915, 1.68229, 1.63915, 1.55288, 1.63915, 1.50975, 1.50975, 1.55288, 1.42348, 1.50975, 1.42348, 1.38034, 1.42348, 1.33721, 1.42348, 1.25093, 1.33721, 1.29407, 1.29407, 1.2078, 1.29407, 1.16466, 1.25093, 1.16466, 1.16466, 1.16466, 1.16466, 1.12153, 1.12153, 1.12153, 1.07839, 1.12153, 1.03526, 1.07839, 1.03526, 1.07839, 0.99212, 0.99212, 1.03526, 0.99212, 0.99212, 0.948984, 0.99212, 0.905849, 0.99212, 0.905849, 0.948984, 0.948984, 0.948984, 0.905849, 0.948984, 0.948984, 0.99212, 0.948984, 1.07839, 1.2078, 1.38034, 1.76856, 2.4156, 3.40772, 5.34882, 7.9801, 13.5877, 20.1875, 28.4264, 36.4928, 37.6143, 33.5164, 22.2149, 8.23891, -5.82331, -21.0933, -23.5952, 8.41145, 49.0884, -0.126037]

plt.figure()
plt.plot(fbar)
plt.figure()
plt.plot(fbar_prime)
plt.figure()
plt.plot(fbar_double_prime)
plt.show()

ffprim = []
pprim = []
rzbbs = []
with open("fpol_pres_ffprim_pprime.data", 'r') as fid:
    for line in fid:
        out = line.split(" ")
        ffprim.append(eval(out[2]))
        pprim.append(eval(out[3]))

with open("separated_file.data", 'r') as fid:
    for line in fid:
        if "rbbbs" in line:
            out = fid.readline()[:-2].split(" ")
            for num in out:
                rzbbs.append(eval(num))
                
rzbbs = np.array(rzbbs)
rbbbs = rzbbs[::2]
zbbbs = rzbbs[1::2]
plt.figure()
plt.plot(rbbbs, zbbbs)
plt.show()
breakpoint()
        
plt.figure()
plt.plot(ffprim)
plt.ylabel("ffprime")
plt.xlabel("psi_N")
plt.figure()
plt.plot(pprim)
plt.ylabel("pprime")
plt.xlabel("psi_N")
plt.show()

