# Dr. Andrea Antonello, May 2020. For any question, write to andrea@automata.tech 
from tools.quat_tools import *
from pytransform3d.rotations import *


tolerance = 0.000000000001

# Quaternion conjugate
q = [0.7071, 0.7071, 0.0, 0.0]
q_conj = quat_conj(q)
print('Quaternion conjugate is', q_conj)

# Quaternion norm/normalize
q = [7071, 7071, 0.0, 0.0]
q_norm = quat_norm(q)
print('Original norm is ', q_norm)
q_norm = quat_norm(quat_normalize(q))
print('Normalized norm is ', q_norm)

# Quaternion multiply
q0 = [0, 1, 0, 0]
q1 = [0, 0, 1, 0]
q_mult = quat_multiply(q0, q1)
print('Quaternion multiplication result is ', q_mult)

# Quaternion <-> rotation matrix
rotm = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
q = rotm_to_quat(rotm)
print('Quaternion is', q)
rotm_calc = quat_to_rotm(q)
assert (np.square(np.subtract(rotm, rotm_calc)).mean()) < tolerance

# Quaternion <-> axis angle
angle = 0.12345
axis = [1, 2, 3]
axis_norm = vect_normalize(axis)
print('Axis normalized is', axis_norm)
q = axisangle_to_quat(axis_norm, angle)
print('Quaternion is', q)
axis_calc, angle_calc = quat_to_axisangle(q)
assert (np.square(np.subtract(vect_normalize(axis), axis_calc)).mean()) < tolerance
assert (np.square(np.subtract(angle, angle_calc)).mean()) < tolerance

# Quaternion <-> roll, pitch, yaw (Euler 'ZXY')
roll = 0.3
pitch = 0.2
yaw = 0.1
q = rpy_to_quat(roll, pitch, yaw)
print('Quaternion is', q)
roll_calc, pitch_calc, yaw_calc = quat_to_rpy(q)
assert (np.square(np.subtract([roll, pitch, yaw], [roll_calc, pitch_calc, yaw_calc])).mean()) < tolerance


# Verify pytransform3d package (referred to as STD in the following)
# Quaternion <-> rotation matrix
q = [-70, 30, 23, 10]
rotm = quat_to_rotm(q)
rotm_STD = matrix_from_quaternion(q)
q_STD = quaternion_from_matrix(rotm_STD)
assert (np.square(np.subtract(rotm, rotm_STD)).mean()) < tolerance
assert (np.square(np.subtract(quat_normalize(np.dot(np.sign(q[0]), q)), np.dot(np.sign(q_STD[0]), q_STD))).mean()) < tolerance
assert (np.square(np.subtract(rotm, rotm_STD)).mean()) < tolerance

# Quaternion <-> axis angle
vector = [1, -2, 3]
angle = -1.2
vect_ang = np.hstack((vector, angle))
q = axisangle_to_quat(vector, angle)
q_STD = quaternion_from_axis_angle(vect_ang)
vect_ang_STD = axis_angle_from_quaternion(q_STD)
assert (np.square(np.subtract(np.dot(np.sign(q[0]), q), np.dot(np.sign(q_STD[0]), q_STD))).mean()) < tolerance
assert (np.square(np.subtract(np.sign(angle)*vect_normalize(vector), vect_ang_STD[:3])).mean()) < tolerance
assert (np.square(np.subtract(abs(angle), vect_ang_STD[3])).mean()) < tolerance

# Quaternion <-> Euler angles
roll = 0.7
pitch = 0.3
yaw = 0.1
q = rpy_to_quat(roll, pitch, yaw)
rotm = quat_to_rotm(q)

rotm_STD = matrix_from_euler_zyx([-yaw, -pitch, -roll])
q_STD = quaternion_from_matrix(rotm_STD)
rotm_inv = matrix_from_quaternion(q_STD)
ang_STD = -euler_zyx_from_matrix(rotm_inv)
assert (np.square(np.subtract(np.dot(np.sign(q[0]), q), np.dot(np.sign(q_STD[0]), q_STD))).mean()) < tolerance
assert (np.square(np.subtract(rotm, rotm_STD)).mean()) < tolerance
assert (np.square(np.subtract([yaw, pitch, roll], ang_STD)).mean()) < tolerance



# Quaternion <-> Euler angles - GIMBAL LOCK
roll = 0.3
pitch = -np.pi/2
yaw = 0.1
q = rpy_to_quat(roll, pitch, yaw)
rotm = quat_to_rotm(q)

rotm_STD = matrix_from_euler_zyx([-yaw, -pitch, -roll])
q_STD = quaternion_from_matrix(rotm_STD)
rotm_inv = matrix_from_quaternion(q_STD)
ang_STD = -euler_zyx_from_matrix(rotm_inv)
if any(np.isnan(ang_STD)):
    raise Exception('Gimbal lock encountered')
else:
    assert (np.square(np.subtract(np.dot(np.sign(q[0]), q), np.dot(np.sign(q_STD[0]), q_STD))).mean()) < tolerance
    assert (np.square(np.subtract(rotm, rotm_STD)).mean()) < tolerance
    assert (np.square(np.subtract(wrap_to_pi([yaw, pitch, roll]), wrap_to_pi(ang_STD))).mean()) < tolerance