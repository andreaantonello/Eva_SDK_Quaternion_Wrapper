# Visualization class
import pyvista as pv
from tools.transform_eva import *
from pytransform3d.rotations import *


class PlotSTL:
    def __init__(self, eva_model, q):
        np.set_printoptions(suppress=True)
        self.q = q
        self.eva_model = eva_model
        self.T_tcp = self._find_tcp_transform()

    def _import_stl(self):
        # Read the STL using numpy-stl
        base = "STL/base_ASS.stl"
        link1 = "STL/link1_ASS.stl"
        link2 = "STL/link2_ASS.stl"
        link3 = "STL/link3_ASS.stl"
        link4 = "STL/link4_ASS.stl"
        link5 = "STL/link5_ASS.stl"
        link6 = "STL/link6_ASS.stl"

        # Create the meshes
        self.mesh = {}
        self.mesh['0'] = pv.read(base)
        self.mesh['1'] = pv.read(link1)
        self.mesh['2'] = pv.read(link2)
        self.mesh['3'] = pv.read(link3)
        self.mesh['4'] = pv.read(link4)
        self.mesh['5'] = pv.read(link5)
        self.mesh['6'] = pv.read(link6)
        # Scale joints from mm to m
        for joint in range(0, 7):
            self.mesh[str(joint)].points = self.mesh[str(joint)].points/1000

    def _transform_stl(self):
        self._import_stl()
        tran_local = CreateTransform(self.eva_model)
        for joint in range(1, 7):
            tran_matrix = tran_local.transform_single_joint(self.q, joint-1)
            self.mesh[str(joint)].transform(np.array(tran_matrix))
        self.Tee = tran_local.transform_ee_plate(self.q)

    def _find_tcp_transform(self):
        # Find transformation matrix between end-effector and TCP end
        self._transform_stl()
        yaw = self.eva_model['EVA']['tcp']['angles']['y']
        pitch = self.eva_model['EVA']['tcp']['angles']['p']
        roll = self.eva_model['EVA']['tcp']['angles']['r']
        ypr = [-yaw, -pitch, -roll]
        self.tcp_offset = np.array([[self.eva_model['EVA']['tcp']['offset']['x']],
                                    [self.eva_model['EVA']['tcp']['offset']['y']],
                                    [self.eva_model['EVA']['tcp']['offset']['z']],
                                    [1]])

        rot_tcp = matrix_from_euler_zyx(ypr)
        tcp_transform = np.vstack((rot_tcp, [0, 0, 0]))
        tcp_transform = np.hstack((tcp_transform, self.tcp_offset))
        return tcp_transform

    @staticmethod
    def _plot_frame(plot, frame, size=0.1):
        scale = 1/size
        center = [row[3] for row in frame[0:3]]
        x = [1, 0, 0]
        y = [0, 1, 0]
        z = [0, 0, 1]
        x_abs = frame[0:3, 0:3].dot(np.transpose(x))
        y_abs = frame[0:3, 0:3].dot(np.transpose(y))
        z_abs = frame[0:3, 0:3].dot(np.transpose(z))

        x_axis = pv.Arrow(np.zeros(3), x_abs[0:3], shaft_radius=0.02, tip_radius=0.05)
        y_axis = pv.Arrow(np.zeros(3), y_abs[0:3], shaft_radius=0.02, tip_radius=0.05)
        z_axis = pv.Arrow(np.zeros(3), z_abs[0:3], shaft_radius=0.02, tip_radius=0.05)

        x_axis.points /= scale
        y_axis.points /= scale
        z_axis.points /= scale

        x_axis.points = x_axis.points + center
        y_axis.points = y_axis.points + center
        z_axis.points = z_axis.points + center

        plot.add_mesh(x_axis, show_edges=False, color='red')
        plot.add_mesh(y_axis, show_edges=False, color='green')
        plot.add_mesh(z_axis, show_edges=False, color='blue')
        return plot

    def _plot_tcp(self, plot):
        plot = self._plot_frame(plot, self.Tee, 0.05)
        T_tcp_absolute = self.Tee.dot(self.T_tcp)
        plot = self._plot_frame(plot, T_tcp_absolute)
        pos_tcp_absolute = [row[3] for row in T_tcp_absolute[0:3]]
        sphere = pv.Sphere(radius=0.01, center=pos_tcp_absolute)
        plot.add_mesh(sphere, show_edges=False, color='black')

        # Plot TCP simplified geometry
        pointa = [row[3] for row in self.Tee]
        pointb = (self.Tee.dot([0, 0, self.tcp_offset[2], 1]))
        pointc = pos_tcp_absolute

        line1 = pv.Line(pointa[0:3], pointb[0:3], resolution=10)
        line2 = pv.Line(pointb[0:3], pointc[0:3], resolution=1)
        plot.add_mesh(line1, show_edges=False, color='black')
        plot.add_mesh(line2, show_edges=False, color='black')
        return plot

    def _plot_all_frames(self, plot):
        tran_local = CreateTransform(self.eva_model)
        T_prog = tran_local.transform_all(self.q)
        for frame in T_prog:
            plot = self._plot_frame(plot, frame, size=0.15)

    @staticmethod
    def plot_sphere(plot, pos, color_user='blue'):
        sphere = pv.Sphere(radius=0.01, center=pos)
        plot.add_mesh(sphere, show_edges=False, color=color_user)
        return plot

    def plot_pose(self, plot, tcp=True, frames=False):
        # Plot robot
        for joint in range(0, 7):
            plot.add_mesh(self.mesh[str(joint)], show_edges=False, color='gray')
        # Plot TCP
        if tcp:
            plot = self._plot_tcp(plot)
        if frames:
            self._plot_all_frames(plot)
        plot.add_floor()
        return plot








