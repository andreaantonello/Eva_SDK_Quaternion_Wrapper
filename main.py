# Dr. Andrea Antonello, May 2020. For any question, write to andrea@automata.tech 
from evasdk import Eva
from tools.transform_eva import *
from tools.plot_eva import *
from config.config_manager import *


if __name__ == "__main__":
    # Load model parameters from YAML and connect to robot
    eva_model = load_use_case_config()
    eva = Eva(eva_model['EVA']['comm']['host'], eva_model['EVA']['comm']['token'])
    q_initial = eva_model['EVA']['tcp']['initial_pos']

    # Initialize transforms, plotter and straightener. Load plotter
    transform = Transformer(eva, eva_model)
    plotter = PlotterEva(eva, eva_model, transform)

    # Straighten head
    q_straightened = transform.straighten_head(q_initial, axis6_constant=True, compact=True)

    # Plot original TCP frame
    plot = pv.Plotter(shape=(1, 2))
    plot = plotter.plot_pose(plot, q_initial, tcp=True, frames=True, subplot=[0, 0], title='Original pose')
    plot = plotter.plot_pose(plot, q_straightened, tcp=True, frames=True, subplot=[0, 1], title='Straightened pose')
    plot.show(full_screen=True)

    exit()

    # Given the initial position, rotate by yaw, pitch roll angles
    # around the TCP tip and find the final position.
    q_rotated = transform.transform_tcp_to_joint_angles(q_straightened, yaw=-0.2, pitch=-0.5, roll=-0.2)

    # Given the initial and final position, find the yaw, pitch roll angles
    # around the TCP tip, along with the XYZ location of the TCP.
    ypr, pos_tcp, pos_ee = transform.transform_joint_angles_to_tcp(q_straightened, q_rotated)

    # Plot updated position
    plot = pv.Plotter()
    plot = plotter.plot_pose(plot, q_rotated, tcp=True, frames=True)
    plot.show(full_screen=True)
