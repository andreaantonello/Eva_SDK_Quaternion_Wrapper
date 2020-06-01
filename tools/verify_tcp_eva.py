# Transformation class
from tools.plot_eva import *


class UseCaseSetup:
    def __init__(self, eva, eva_model):
        np.set_printoptions(suppress=True)
        self.eva = eva
        self.eva_model = eva_model
        self.tran = CreateTransform(self.eva, self.eva_model)
        self.plot_eva = PlotEva(self.eva, self.eva_model)

    def run(self):
        if not yes_no_answer('Have you already verified the TCP position? [yes/no]\n'):
            q_straight = self.confirm_tcp_config()
        else:
            q_straight = self._straighten_head()
        tcp_transform = self.tran.transform_ee_plate_to_tcp()
        return q_straight, tcp_transform

    def confirm_tcp_config(self):
        # Extract initial position and straighten it
        q_straight = self._straighten_head()

        # Create plotter
        plot = pv.Plotter()
        plot = self.plot_eva.plot_pose(plot, q_straight)
        plot.show()
        if not yes_no_answer('Is the initial pose and TCP configuration correct?\n'):
            raise Exception('Program aborted. Verify joint and TCP definition.')

        return q_straight

    def _straighten_head(self):
        q = self.eva_model['EVA']['tcp']['initial_pos']
        fk = self.eva.calc_forward_kinematics(q)
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
        ik_result = self.eva.calc_inverse_kinematics(q, pos_json, orient_json)
        if 'success' not in ik_result['ik']['result']:
            raise Exception('Inverse kinematics failed')
        q_straight = ik_result['ik']['joints']
        return q_straight


def yes_no_answer(question):
    while True:
        yes_no = input(question)
        if yes_no in ['y', 'Y', 'yes', 'Yes']:
            return True
        elif yes_no in ['n', 'N', 'no', 'No']:
            return False
        else:
            print('Insert answer in correct format. Retry...')
