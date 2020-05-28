# Transformation class
import numpy as np


class CreateTransform:
    def __init__(self, eva_model):
        np.set_printoptions(suppress=True)
        self.alp = eva_model['EVA']['dh']['alpha']
        self.a = eva_model['EVA']['dh']['a']
        self.d = eva_model['EVA']['dh']['d']
        self.off = eva_model['EVA']['dh']['offset']
        self.ee_plate = eva_model['EVA']['dh']['ee_plate']

    def _transform_list(self, q):
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

    def transform_all(self, q):
        _, _, Tprog = self._transform_list(q)
        return Tprog

    def transform_tcp(self, q, joint):
        _, _, Tprog = self._transform_list(q)
        return Tprog[joint]

    def transform_single_joint(self, q, joint):
        if not 0 <= joint <= 5:
            raise Exception('Joint outside range')
        _, _, Tprog = self._transform_list(q)
        return Tprog[joint]

    def transform_ee_plate(self, q):
        _, Tee, _ = self._transform_list(q)
        return Tee

    def forward_kin(self, q):
        fk = []
        _, Tee, _ = self._transform_list(q)
        fk['pos'] = [Tee[0, 3], Tee[1, 3], Tee[2, 3]]
        fk['quat'] = self.rot2quat(Tee)
        return fk

    def joints_from_tcp(self, tcp_fixed_frame, pos, quat):
        pos = {'x': pos[1], 'y': pos[2], 'z': pos[3]}
        quat = {'w': quat[0], 'x': quat[1], 'y': quat[2], 'z': quat[3]}