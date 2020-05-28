# Dr. Andrea Antonello, May 2020. For any question, write to andrea@automata.tech 
from evasdk import Eva, EvaError
from stl.mesh import Mesh
import vtkplotlib as vpl
from tools.plot_eva import *
from tools.verify_tcp_eva import *
from config.config_manager import *
from pytransform3d.rotations import *

# Load model parameters from YAML and initialize transform class
eva_model = load_use_case_config()
tran = CreateTransform(eva_model)

# Connect to robot
eva = Eva(eva_model['EVA']['comm']['host'], eva_model['EVA']['comm']['token'])

# TCP calibration
if not yes_no_answer('Have you already verified the TCP position? [yes/no]\n'):
    confirm_tcp_config(eva, eva_model)
exit()



res = tran.transform_single_joint([0, 0, 0, 0, 0, 0], 0)
res2 = tran.transform_ee_plate([0, 0, 0, 0, 0, 0])

# Load plotter
plot = pv.Plotter()

# Connection to office robot
host_ip = 'http://office.automata.tech:4242'
token = '3becdcbe6622bc87e31556bcf2cbcc0fe08a739b'
eva = Eva(host_ip, token)

# Check for the initial position and TCP location
q = [0, 0.6, -2, 0, -1.5, 0]
fk = eva.calc_forward_kinematics(q)
print(fk)
orient = {'w': 0, 'x': 0, 'y': 1, 'z': 0}
pos = fk['position']

q_ik = eva.calc_inverse_kinematics(q, fk['position'], orient)
q = q_ik['ik']['joints']

plotEva = PlotSTL(eva_model, q)
plot = plotEva.plot_pose(plot, tcp=True, frames=False)
T_tcp = plotEva.T_tcp

plot.show()


exit()
fk = eva.calc_forward_kinematics(q)
T_ee = tran.transform_ee_plate(q)
T_tcp_abs = T_ee.dot(T_tcp)

pos = [fk['position']['x'], fk['position']['y'], fk['position']['z']]
quat = [0, 0, 1, 0]

pos_card = [row[3] for row in T_tcp_abs[0:3]]
plot = plotEva.plot_sphere(plot, pos_card, 'red')


rot_tcp = matrix_from_euler_zyx([0, 0, 0])
# print(rot_tcp)

rot_tcp = np.vstack((rot_tcp, [0, 0, 0]))
rot_tcp = np.hstack((rot_tcp, [[0], [0], [0], [1]]))
T_tcp_abs_rotated = T_tcp_abs.dot(rot_tcp)

T_ee = T_tcp_abs_rotated.dot(np.linalg.inv(T_tcp))


quat_ee = quaternion_from_matrix(T_ee[0:3, 0:3])
quat_ee = {'w': quat_ee[0], 'x': quat_ee[1], 'y': quat_ee[2], 'z': quat_ee[3]}
pos_ee = [row[3] for row in T_ee[0:3]]
pos_ee = {'x': pos_ee[0], 'y': pos_ee[1], 'z': pos_ee[2]}

q_ik = eva.calc_inverse_kinematics(q, pos_ee, quat_ee)
q = q_ik['ik']['joints']
# print('q new', q)

plotEva = PlotSTL(eva_model, q)
plot = plotEva.plot_pose(plot, tcp=True, frames=False)


fk = eva.calc_forward_kinematics(q)
print('Orientazione mona', fk['orientation'])


# App to rotate by a set amount of degrees






exit()
base = "STL/base_ASS.stl"
link1 = "STL/link1_ASS.stl"
link2 = "STL/link2_ASS.stl"

# Read the STL using numpy-stl
mesh_base = Mesh.from_file(base)
mesh_link1 = Mesh.from_file(link1)
mesh_link2 = Mesh.from_file(link2)

mesh_base.transform(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
print(mesh_base[0])

# Plot the mesh
vpl.mesh_plot(mesh_base)
vpl.mesh_plot(mesh_link1)
vpl.mesh_plot(mesh_link2)

# Show the figure
vpl.show()


# Connection to head-less robot in the office
host_ip = 'http://office.automata.tech:4242'
token = '3becdcbe6622bc87e31556bcf2cbcc0fe08a739b'
eva = Eva(host_ip, token)
print(eva.data_servo_positions())


exit()
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