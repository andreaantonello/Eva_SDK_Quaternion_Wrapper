# Dr. Andrea Antonello, May 2020. For any question, write to andrea@automata.tech 
from evasdk import Eva, EvaError
from stl.mesh import Mesh
import vtkplotlib as vpl
from tools.plot_eva import *
from tools.verify_tcp_eva import *
from config.config_manager import *
from pytransform3d.rotations import *


# Load model parameters from YAML
eva_model = load_use_case_config()

# Connect to robot
eva = Eva(eva_model['EVA']['comm']['host'], eva_model['EVA']['comm']['token'])

# Initialize classes
tran = CreateTransform(eva, eva_model)
setup = UseCaseSetup(eva, eva_model)
plotEva = PlotEva(eva, eva_model)

# Obtain TCP transform
q_corrected, tcp_transform = setup.run()

# Load plotter
plot = pv.Plotter()

# Plot original TCP frame
tcp_init = tran.transform_tcp(q_corrected)
plotEva.plot_frame(plot, tcp_init)

# Change yaw, pitch roll around TCP
q_rotated = tran.transform_tcp_to_joint_angles(q_corrected, tcp_transform, yaw=-0.2, pitch=-0.5, roll=-0.2)

# Plot updated position
plot = plotEva.plot_pose(plot, q_rotated, tcp=True, frames=False)
plot.show()
exit()

