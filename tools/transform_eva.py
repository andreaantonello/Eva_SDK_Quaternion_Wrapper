# Transformation class
import pytransform3d.rotations as pyrot  # type: ignore
import numpy as np
import copy
np.set_printoptions(suppress=True)


class Transformer:
    def __init__(self, eva, eva_model):
        self.eva = eva
        self.eva_model = eva_model
        self.tcp = self.eva_model['EVA']['tcp']
        self.alp = self.eva_model['EVA']['dh']['alpha']
        self.a = self.eva_model['EVA']['dh']['a']
        self.d = self.eva_model['EVA']['dh']['d']
        self.off = self.eva_model['EVA']['dh']['offset']
        self.ee_plate = self.eva_model['EVA']['dh']['ee_plate']

    def _transform_list(self, q):
        """ This function returns all the 4x4 transform matrices for a given joint angles set (q).
        NOTE: Tprog is obtained as successive multiplication of Tall transforms, i.e. Tprog[2]= Tall[0]*Tall[1]*Tall[2])
        NOTE: Tee is obtained as successive multiplication of ALL Tall transforms.
        :param q: joint angles
        :return: Tall: transform from joint i-1 to i, for i=0:5
        :return: Tprog: transform from the base frame to all joints
        :return: Tee: transform from the base frame to the end-effector frame
        """
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
        """ This function returns the 4x4 transform matrix from the base frame to the end-effector frame by invoking
        the transform_list method
        :param q: joint angles
        :return: Tee: transform from the base frame to the end-effector frame
        """
        _, Tee, _ = self._transform_list(q)
        return Tee

    def transform_all_joints(self, q):
        """ This function returns the 4x4 transform matrices from the base frame to each joint by invoking
        the transform_list method
        :param q: joint angles
        :return: Tprog: transform from the base frame to all joints
        """
        # Find all progressive transformation matrices (T01, T12, ..., T56)
        _, _, Tprog = self._transform_list(q)
        return Tprog

    def transform_single_joint(self, q, joint):
        """ This function returns the 4x4 transform matrix from the base frame to joint i by invoking
        the transform_list method
        :param q: joint angles
        :param joint:
        :return: Tprog[joint]: transform from the base frame to joint i
        """
        # Find a specific progressive transformation matrix
        if not 0 <= joint <= 5:
            raise Exception('Joint outside range')
        Tprog = self.transform_all_joints(q)
        return Tprog[joint]

    @staticmethod
    def transform_ee_plate_to_tcp(tcp):
        """ This function returns the 4x4 transform matrix from the end-effector plate to the TCP
        :param tcp: dict containing offset and angle values for the TCP
        :return: tcp_transform: 4x4 transform matrix
        """
        # Find transformation matrix between end-effector and TCP
        yaw = tcp['angles']['y']
        pitch = tcp['angles']['p']
        roll = tcp['angles']['r']
        ypr = [-yaw, -pitch, -roll]
        tcp_offset = np.array([[tcp['offset']['x']],
                               [tcp['offset']['y']],
                               [tcp['offset']['z']],
                               [1]])

        tcp_rotation = pyrot.matrix_from_euler_zyx(ypr)
        tcp_transform = np.vstack((tcp_rotation, [0, 0, 0]))
        tcp_transform = np.hstack((tcp_transform, tcp_offset))
        return tcp_transform

    def transform_base_to_tcp(self, q):
        """ This function returns the 4x4 transform matrix from base to the TCP
        :param q: joint angles
        :return: tcp_transform_abs: 4x4 transform matrix
        """
        # Find transformation matrix between base and TCP
        tcp_transform = self.transform_ee_plate_to_tcp(self.tcp)
        ee_transform = self.transform_base_to_ee_plate(q)
        tcp_transform_abs = ee_transform.dot(tcp_transform)
        return tcp_transform_abs

    @staticmethod
    def _matrix_from_ypr(ypr):
        ypr_rotation = pyrot.matrix_from_euler_zyx(ypr)
        return ypr_rotation

    @staticmethod
    def _ypr_from_matrix(ypr_rotation):
        ypr = pyrot.euler_zyx_from_matrix(ypr_rotation)
        return ypr

    def transform_tcp_to_joint_angles(self, q_init, yaw=0, pitch=0, roll=0, tcp=None):
        """ This function rotates the joint angles q_init by the yaw, pitch, and roll angles
        provided, given an input tcp position (tcp_transform). It outputs the joint angles
        corresponding to this new configuration.
        :param q_init: joint angles (initial position)
        :param yaw: in radians
        :param pitch: in radians
        :param roll: in radians
        :param tcp: dict containing offset and angle values for the TCP [optional]
        :return: q_fin: joint angles (rotated position)
        """
        ypr = [yaw, pitch, roll]
        ypr_rotation = self._matrix_from_ypr(ypr)
        ee_transform = self.transform_base_to_ee_plate(q_init)
        if tcp is None:  # Use stock TCP
            tcp_transform = self.transform_ee_plate_to_tcp(self.tcp)
        else:
            tcp_transform = self.transform_ee_plate_to_tcp(tcp)
        tcp_transform_abs = ee_transform.dot(tcp_transform)

        ypr_rotation = np.vstack((ypr_rotation, [0, 0, 0]))
        ypr_transform = np.hstack((ypr_rotation, [[0], [0], [0], [1]]))
        tcp_transform_abs_ypr = tcp_transform_abs.dot(ypr_transform)

        T_ee_ypr = tcp_transform_abs_ypr.dot(np.linalg.inv(tcp_transform))

        quat_ee = pyrot.quaternion_from_matrix(T_ee_ypr[0:3, 0:3])
        quat_ee = {'w': quat_ee[0], 'x': quat_ee[1], 'y': quat_ee[2], 'z': quat_ee[3]}
        pos_ee = [row[3] for row in T_ee_ypr[0:3]]
        pos_ee = {'x': pos_ee[0], 'y': pos_ee[1], 'z': pos_ee[2]}

        ik_result = self.eva.calc_inverse_kinematics(q_init, pos_ee, quat_ee)
        if 'success' not in ik_result['ik']['result']:
            raise Exception('Inverse kinematics failed, guess is too far.')
        q_fin = ik_result['ik']['joints']
        return q_fin

    def transform_joint_angles_to_tcp(self, q_init, q_fin, tcp=None):
        """ This function computes the xyz positions, the yaw, pitch, and roll angles
        given an input position (q_init) and a rotated joint position (q_fin).
        It requires the input of the TCP configuration (tcp_transform).
        :param q_init: joint angles (initial position)
        :param q_fin: joint angles (rotated position)
        :param tcp: dict containing offset and angle values for the TCP [optional]
        :return: ypr_dict: yaw, pitch and roll angles
        :return: pos_tcp: cartesian position of TCP
        :return: pos_ee: cartesian position of end-effector
        """
        if tcp is None:  # Use stock TCP
            tcp_transform = self.transform_ee_plate_to_tcp(self.tcp)
        else:
            tcp_transform = self.transform_ee_plate_to_tcp(tcp)
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

    def straighten_head(self, q_initial=None, axis6_constant=False):
        if q_initial is None:
            q_initial = self.eva_model['EVA']['tcp']['initial_pos']
        fk = self.eva.calc_forward_kinematics(q_initial)
        orient = [float(fk['orientation']['w']),
                  float(fk['orientation']['x']),
                  float(fk['orientation']['y']),
                  float(fk['orientation']['z'])]
        # ALERT!! Need to verify the code below - this is still IDEX legacy
        orient_straight = [0, 0, 1, 0]
        print(f'WARNING: Orientation has been straighted with head pointing down!\n'
              f' -> Original quaternion was: {orient} \n',
              f'-> Straightened quaternion is now: {orient_straight} \n')
        pos_json = fk['position']
        orient_json = {'w': orient_straight[0],
                       'x': orient_straight[1],
                       'y': orient_straight[2],
                       'z': orient_straight[3]}

        ik_result = self.eva.calc_inverse_kinematics(q_initial, pos_json, orient_json)
        if 'success' not in ik_result['ik']['result']:
            raise Exception('Inverse kinematics failed')
        q_straightened = ik_result['ik']['joints']
        if axis6_constant is True:
            q_straightened_axis6_constant = copy.copy(q_straightened)
            q_straightened_axis6_constant[5] = copy.copy(q_initial[5])
            fk_straightened_axis6_constant = self.eva.calc_forward_kinematics(q_straightened_axis6_constant)
            ik_result_straightened = self.eva.calc_inverse_kinematics(q_straightened_axis6_constant,
                                                                      fk_straightened_axis6_constant['position'],
                                                                      fk_straightened_axis6_constant['orientation'])
            if 'success' not in ik_result['ik']['result']:
                raise Exception('Inverse kinematics failed')
            q_straightened_axis6_constant = ik_result_straightened['ik']['joints']
            return q_straightened_axis6_constant
        return q_straightened
