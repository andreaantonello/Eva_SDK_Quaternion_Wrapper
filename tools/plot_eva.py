# Visualization class
import pyvista as pv
from pytransform3d.rotations import *
np.set_printoptions(suppress=True)

X = [1, 0, 0]
Y = [0, 1, 0]
Z = [0, 0, 1]


class PlotterEva:
    def __init__(self, eva, eva_model, transform):
        self.eva = eva
        self.eva_model = eva_model
        self.transform = transform
        self.mesh = {}

    def _import_stl(self):
        # Read the STL using numpy-stl
        base = "STL/base.stl"
        card = "STL/card.stl"
        link1 = "STL/link1.stl"
        link2 = "STL/link2.stl"
        link3 = "STL/link3.stl"
        link4 = "STL/link4.stl"
        link5 = "STL/link5.stl"
        link6 = "STL/link6.stl"

        # Create the meshes
        self.mesh['0'] = pv.read(base)
        self.mesh['1'] = pv.read(link1)
        self.mesh['2'] = pv.read(link2)
        self.mesh['3'] = pv.read(link3)
        self.mesh['4'] = pv.read(link4)
        self.mesh['5'] = pv.read(link5)
        self.mesh['6'] = pv.read(link6)
        self.mesh['card'] = pv.read(card)

        # Scale STL from mm to m
        for joint in range(0, 7):
            self.mesh[str(joint)].points = self.mesh[str(joint)].points/1000

    def _transform_stl(self, q):
        self._import_stl()
        self.q = q
        for joint in range(1, 7):
            tran_matrix = self.transform.transform_single_joint(self.q, joint-1)
            self.mesh[str(joint)].transform(np.array(tran_matrix))
        self.Tee = self.transform.transform_base_to_ee_plate(self.q)

    @staticmethod
    def plot_frame(plot, frame, size=0.1):
        scale = 1/size
        center = [row[3] for row in frame[0:3]]
        x_abs = frame[0:3, 0:3].dot(np.transpose(X))
        y_abs = frame[0:3, 0:3].dot(np.transpose(Y))
        z_abs = frame[0:3, 0:3].dot(np.transpose(Z))

        x_axis = pv.Arrow(np.zeros(3), x_abs[0:3], shaft_radius=0.02, tip_radius=0.05)
        y_axis = pv.Arrow(np.zeros(3), y_abs[0:3], shaft_radius=0.02, tip_radius=0.05)
        z_axis = pv.Arrow(np.zeros(3), z_abs[0:3], shaft_radius=0.02, tip_radius=0.05)

        x_axis.points /= scale
        y_axis.points /= scale
        z_axis.points /= scale

        x_axis.points += center
        y_axis.points += center
        z_axis.points += center

        plot.add_mesh(x_axis, show_edges=False, color='red')
        plot.add_mesh(y_axis, show_edges=False, color='green')
        plot.add_mesh(z_axis, show_edges=False, color='blue')
        return plot

    def _plot_tcp(self, plot):
        tcp_transform = self.transform.transform_ee_plate_to_tcp(self.transform.tcp)
        plot = self.plot_frame(plot, self.Tee, 0.05)
        tcp_transform_abs = self.Tee.dot(tcp_transform)
        plot = self.plot_frame(plot, tcp_transform_abs)
        pos_tcp_absolute = [row[3] for row in tcp_transform_abs[0:3]]
        sphere = pv.Sphere(radius=0.01, center=pos_tcp_absolute)
        plot.add_mesh(sphere, show_edges=False, color='black')

        # Plot CARD
        self.mesh['card'].points = self.mesh['card'].points / 1000
        self.mesh['card'].transform(np.array(tcp_transform_abs))
        plot.add_mesh(self.mesh['card'], show_edges=False, color='yellow', opacity=1,
                      ambient=0.3, diffuse=0.3, specular=10.0, specular_power=100.0)

        # Plot TCP simplified geometry
        point_a = [row[3] for row in self.Tee]
        point_b = (self.Tee.dot([0, 0, self.eva_model['EVA']['tcp']['offset']['z'], 1]))
        point_c = pos_tcp_absolute
        line1 = pv.Line(point_a[0:3], point_b[0:3], resolution=1)
        line2 = pv.Line(point_b[0:3], point_c[0:3], resolution=1)
        plot.add_mesh(line1, show_edges=False, color='black')
        plot.add_mesh(line2, show_edges=False, color='black')
        return plot

    def _plot_all_frames(self, plot):
        T_prog = self.transform.transform_all_joints(self.q)
        for frame in T_prog:
            plot = self.plot_frame(plot, frame, size=0.15)

    @staticmethod
    def plot_sphere(plot, pos, color_user='black'):
        sphere = pv.Sphere(radius=0.01, center=pos)
        plot.add_mesh(sphere, show_edges=False, color=color_user)
        return plot

    def plot_pose(self, plot, q, tcp=True, frames=False, subplot=None, title=None):
        self.mesh = {}
        self.eva.calc_pose_valid(q)  # Verify pose validity
        self._transform_stl(q)  # Obtain all STL positions
        if subplot is not None:
            plot.subplot(subplot[0], subplot[1])
        for joint in range(0, 7):  # Plot robot
            plot.add_mesh(self.mesh[str(joint)], show_edges=False, color='white', opacity=1,
                          ambient=0.4, diffuse=0.3, specular=10.0, specular_power=100.0)
        if tcp:  # Plot TCP
            plot = self._plot_tcp(plot)
        if frames:  # Plot frames
            self._plot_all_frames(plot)
        plot.add_floor()
        plot.add_text(title)
        return plot
