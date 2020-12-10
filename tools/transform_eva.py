# Transformation class
from pytransform3d.rotations import *


class CreateTransform:
    def __init__(self, eva, eva_model):
        np.set_printoptions(suppress=True)
        self.eva = eva
        self.tcp = eva_model['EVA']['tcp']
        self.alp = eva_model['EVA']['dh']['alpha']
        self.a = eva_model['EVA']['dh']['a']
        self.d = eva_model['EVA']['dh']['d']
        self.off = eva_model['EVA']['dh']['offset']
        self.ee_plate = eva_model['EVA']['dh']['ee_plate']

    def _transform_list(self, q):
        # Find all transformation matrices for Eva
        Tall = []
        Tprog = []
        Ttcp = np.array([[1, 0, 0, self.ee_plate[0]],
                         [0, 1, 0, self.ee_plate[1]],
                         [0, 0, 1, self.ee_plate[2]],
                         [0, 0, 0, 1]])

        for joint in range(len(q)):
            cosq = np.cos(q[joint])
            sinq = np.sin(q[joint])
            sino = np.sin(self.off[joint])
            coso = np.cos(self.off[joint])
            sina = np.sin(self.alp[joint])
            cosa = np.cos(self.alp[joint])

            Ti = np.array([[cosq*coso - cosa*sinq*sino, -coso*sinq - cosa*cosq*sino, sina*sino, self.a[joint]*coso],
                           [cosq*sino + cosa*coso*sinq, cosa*cosq*coso - sinq*sino, -sina*coso, self.a[joint]*sino],
                           [sina*sinq, cosq*sina, cosa, self.d[joint]],
                           [0, 0, 0, 1]])
            Tall.append(Ti)
        Tprog.append(Tall[0])
        Tprog.append(Tprog[0].dot(Tall[1]))
        Tprog.append(Tprog[1].dot(Tall[2]))
        Tprog.append(Tprog[2].dot(Tall[3]))
        Tprog.append(Tprog[3].dot(Tall[4]))
        Tprog.append(Tprog[4].dot(Tall[5]))
        Tee = Tall[0].dot(Tall[1]).dot(Tall[2]).dot(Tall[3]).dot(Tall[4]).dot(Tall[5]).dot(Ttcp)
        return Tall, Tee, Tprog

    def transform_base_to_ee_plate(self, q):
        # Find transformation matrix between base and end-effector
        _, Tee, _ = self._transform_list(q)
        return Tee

    def transform_ee_plate_to_tcp(self):
        # Find transformation matrix between end-effector and TCP
        yaw = self.tcp['angles']['y']
        pitch = self.tcp['angles']['p']
        roll = self.tcp['angles']['r']
        ypr = [-yaw, -pitch, -roll]
        tcp_offset = np.array([[self.tcp['offset']['x']],
                               [self.tcp['offset']['y']],
                               [self.tcp['offset']['z']],
                               [1]])

        tcp_rotation = matrix_from_euler_zyx(ypr)
        tcp_transform = np.vstack((tcp_rotation, [0, 0, 0]))
        tcp_transform = np.hstack((tcp_transform, tcp_offset))
        return tcp_transform

    def transform_base_to_tcp(self, q):
        # Find transformation matrix between base and TCP
        tcp_transform = self.transform_ee_plate_to_tcp()
        ee_transform = self.transform_base_to_ee_plate(q)
        tcp_transform_abs = ee_transform.dot(tcp_transform)
        return tcp_transform_abs

    def transform_all_joints(self, q):
        # Find all progressive transformation matrices (T01, T12, ..., T56)
        _, _, Tprog = self._transform_list(q)
        return Tprog

    def transform_single_joint(self, q, joint):
        # Find a specific progressive transformation matrix
        if not 0 <= joint <= 5:
            raise Exception('Joint outside range')
        _, _, Tprog = self._transform_list(q)
        return Tprog[joint]

    @staticmethod
    def _matrix_from_ypr(ypr):
        ypr_rotation = matrix_from_euler_zyx(ypr)
        return ypr_rotation

    @staticmethod
    def _ypr_from_matrix(ypr_rotation):
        ypr = euler_zyx_from_matrix(ypr_rotation)
        return ypr

    def transform_tcp_to_joint_angles(self, q_init, tcp_transform, yaw=0, pitch=0, roll=0):
        """ This function rotates the joint angles q_init by the yaw, pitch, and roll angles
        provided, given an input tcp position (tcp_transform). It outputs the joint angles
        corresponding to this new configuration.

        :param q_init: joint angles (initial position)
        :param tcp_transform: 4x4 transformation matrix between end-effector and TCP
        :param yaw: in radians
        :param pitch: in radians
        :param roll: in radians
        :return: q_fin: joint angles (rotated position)
        """
        ypr = [yaw, pitch, roll]
        ypr_rotation = self._matrix_from_ypr(ypr)
        ee_transform = self.transform_base_to_ee_plate(q_init)
        tcp_transform_abs = ee_transform.dot(tcp_transform)

        ypr_rotation = np.vstack((ypr_rotation, [0, 0, 0]))
        ypr_transform = np.hstack((ypr_rotation, [[0], [0], [0], [1]]))
        tcp_transform_abs_ypr = tcp_transform_abs.dot(ypr_transform)

        T_ee_ypr = tcp_transform_abs_ypr.dot(np.linalg.inv(tcp_transform))

        quat_ee = quaternion_from_matrix(T_ee_ypr[0:3, 0:3])
        quat_ee = {'w': quat_ee[0], 'x': quat_ee[1], 'y': quat_ee[2], 'z': quat_ee[3]}
        pos_ee = [row[3] for row in T_ee_ypr[0:3]]
        pos_ee = {'x': pos_ee[0], 'y': pos_ee[1], 'z': pos_ee[2]}

        ik_result = self.eva.calc_inverse_kinematics(q_init, pos_ee, quat_ee)
        if 'success' not in ik_result['ik']['result']:
            raise Exception('Inverse kinematics failed, guess is too far.')
        q_fin = ik_result['ik']['joints']
        return q_fin

    def transform_joint_angles_to_tcp(self, q_init, q_fin, tcp_transform):
        """ This function computes the xyz positions, the yaw, pitch, and roll angles
        given an input position (q_init) and a rotated joint position (q_fin).
        It requires the input of the TCP configuration (tcp_transform).

        :param q_init: joint angles (initial position)
        :param q_fin: joint angles (rotated position)
        :param tcp_transform: 4x4 transformation matrix between end-effector and TCP
        :return: ypr_dict: yaw, pitch and roll angles
        :return: pos_tcp: cartesian position of TCP
        :return: pos_ee: cartesian position of end-effector
        """
        T_ee_ypr = self.transform_base_to_ee_plate(q_fin)
        tcp_transform_abs_ypr = T_ee_ypr.dot(tcp_transform)
        # Find cartesian positions
        pos_tcp = [row[3] for row in tcp_transform_abs_ypr[0:3]]
        pos_ee = [row[3] for row in T_ee_ypr[0:3]]

        # Find yaw, pitch and roll angles
        ee_transform = self.transform_base_to_ee_plate(q_init)
        tcp_transform_abs = ee_transform.dot(tcp_transform)
        ypr_transform = np.linalg.inv(tcp_transform_abs).dot(tcp_transform_abs_ypr)
        ypr_rotation = ypr_transform[0:3, 0:3]
        ypr = self._ypr_from_matrix(ypr_rotation)
        ypr_dict = {'yaw': ypr[0], 'pitch': ypr[1], 'roll': ypr[2]}
        return ypr_dict, pos_tcp, pos_ee
