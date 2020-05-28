# Transformation class
import numpy as np
import pyvista as pv
from tools.plot_eva import *


def confirm_tcp_config(eva, eva_model):
    # Load plotter
    plot = pv.Plotter()
    q = eva_model['EVA']['initial_pos']
    plotEva = PlotSTL(eva_model, q)
    plot = plotEva.plot_pose(plot, tcp=True, frames=False)
    plot.show()
    if not yes_no_answer('Is the initial pose and TCP configuration correct?\n'):
        raise Exception('Program aborted. Verify joint and TCP definition.')
    fk = eva.calc_forward_kinematics(q)
    orient = [float(fk['orientation']['w']),
              float(fk['orientation']['x']),
              float(fk['orientation']['y']),
              float(fk['orientation']['z'])]
    orient_straight = [np.round(x) for x in orient]
    print('WARNING: Orientation has been straighted to the closest, adjacent axis!\n'
          ' -> Original quaternion was:', orient, '\n',
          '-> Straightened quaternion is now:', orient_straight, '\n')
    pos_json = fk['position']
    orient_json = {'w': orient_straight[0],
                   'x': orient_straight[1],
                   'y': orient_straight[2],
                   'z': orient_straight[3]}
    ik_result = eva.calc_inverse_kinematics(q, pos_json, orient_json)
    if 'success' not in ik_result['ik']['result']:
        raise Exception('Inverse kinematics failed')
    q_start = ik_result['ik']['joints']

    return q_start, plotEva.T_tcp

def compute_tcp_frame(eva, q):
    fk = eva.calc_forward_kinematics(q)
    orient = [float(fk['orientation']['w']),
              float(fk['orientation']['x']),
              float(fk['orientation']['y']),
              float(fk['orientation']['z'])]
    orient_straight = [np.round(x) for x in orient]
    print('WARNING: Orientation has been straighted to the closest, adjacent axis!\n'
          ' -> Original quaternion was:', orient, '\n',
          '-> Straightened quaternion is now:', orient_straight, '\n')
    pos_json = fk['position']
    orient_json = {'w': orient_straight[0],
                   'x': orient_straight[1],
                   'y': orient_straight[2],
                   'z': orient_straight[3]}
    ik_result = eva.calc_inverse_kinematics(q, pos_json, orient_json)
    if 'success' not in ik_result['ik']['result']:
        raise Exception('Inverse kinematics failed')
    q_start = ik_result['ik']['joints']

    return q_start, plotEva.T_tcp


def yes_no_answer(question):
    while True:
        yes_no = input(question)
        if yes_no in ['y', 'Y', 'yes', 'Yes']:
            return True
        elif yes_no in ['n', 'N', 'no', 'No']:
            return False
        else:
            print('Insert answer in correct format. Retry...')



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