from fileinput import filename
from time import sleep
from typing import Tuple
import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import matplotlib.lines as mlines
import math
import argparse
import glob
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import os
import json
from scipy.spatial.transform import Rotation as R
from sympy import Point, Line
from sympy.solvers import solve
from sympy import Symbol
from sympy import Point3D
from sympy.geometry import Line3D

fig, (HLAax, VLAax, SAax) = None, (None, None, None)


class ItkImage:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.augment_mhd_file()
        self.load()

    def augment_mhd_file(self):
        new_content = ""
        with open(self.filename, 'r') as fp:
            for line in fp.read().splitlines():
                if "TransformMatrix" in line:
                    line = "TransformMatrix = 1 0 0 0 1 0 0 0 1"
                elif "AnatomicalOrientation" in line:
                    line = "AnatomicalOrientation = LPS"
                new_content += line + "\n"
        with open(self.filename, 'w') as fp:
            fp.write(new_content)

    def load(self) -> None:
        self.image = sitk.ReadImage(self.filename, imageIO="MetaImageIO")  # TODO generalize for other formats
        self.refresh()

    def refresh(self) -> None:
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        self.ct_scan = sitk.GetArrayFromImage(self.image)
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        self.origin = np.array(list(reversed(self.image.GetOrigin())))  # TODO handle different rotations
        # Read the spacing along each dimension
        self.spacing = np.array(list(reversed(self.image.GetSpacing())))

    def resolution(self) -> Tuple[int, int, int]:
        return (self.image.GetWidth(), self.image.GetHeight(), self.image.GetDepth())

    def resample(self, transform):
        """
        This function resamples (updates) an image using a specified transform
        :param image: The sitk image we are trying to transform
        :param transform: An sitk transform (ex. resizing, rotation, etc.
        :return: The transformed sitk image
        """
        reference_image = self.image
        interpolator = sitk.sitkBSpline
        default_value = 0
        return sitk.Resample(self.image, reference_image, transform,
                             interpolator, default_value)

    def get_center(self):
        """
        This function returns the physical center point of a 3d sitk image
        :param img: The sitk image we are trying to find the center of
        :return: The physical center point of the image
        """
        width, height, depth = self.image.GetSize()
        p = self.image.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                                     int(np.ceil(height/2)),
                                                     int(np.ceil(depth/2))))
        print("center", p)
        return p

    def rotation3d(self, theta_x, theta_y, theta_z):
        """
        This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
        respectively
        :param image: An sitk MRI image
        :param theta_x: The amount of degrees the user wants the image rotated around the x axis
        :param theta_y: The amount of degrees the user wants the image rotated around the y axis
        :param theta_z: The amount of degrees the user wants the image rotated around the z axis
        :param show: Boolean, whether or not the user wants to see the result of the rotation
        :return: The rotated image
        """
        self.load()
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)
        euler_transform = sitk.Euler3DTransform(self.get_center(), theta_x, theta_y, theta_z, (0, 0, 0))
        self.image = self.resample(euler_transform)
        self.refresh()


class VolumeImage:

    def eventSetup(self):
        def onScroll(event):
            if selected_axis is not self.ax:
                return

            if event.button == "up":
                self.index += 1
            if event.button == "down":
                self.index -= 1
            self.index = 0 if self.index < 0 else (len(self.image.ct_scan) - 1 if self.index > len(self.image.ct_scan) else self.index)
            self.ax.set_title(f"Slice: {self.index}")
            self.ax_data.set_data(self.image.ct_scan[self.index])
            self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('scroll_event', onScroll)

    def setIndex(self, index: int):
        self.index = index

    def redraw(self):
        self.ax_data.set_data(self.image.ct_scan[self.index])
        self.fig.canvas.draw_idle()


class PlotPlaneSelect(VolumeImage):
    def __init__(self, image: ItkImage, ax, onSetPlane=None, title="PlaneSelect") -> None:
        self.image = image
        self.fig = fig
        self.ax = ax
        self.index = int(len(self.image.ct_scan)/2)
        self.plane = None
        self.patch = None
        self.onSetPlane = onSetPlane
        self.ax_data = self.ax.imshow(self.image.ct_scan[self.index], cmap='gray')
        self.eventSetup()

        self.selectedLine = (None, None)
        self.pressed = False

        def onButtonPress(event):
            if selected_axis is not self.ax:
                return

            self.pressed = True
            self.selectedLine = ((event.xdata, event.ydata), None)

        def onButtonRelease(event):
            if selected_axis is not self.ax:
                return

            self.pressed = False
            self.selectedLine = (self.selectedLine[0], (event.xdata, event.ydata))
            ((x1, y1), (x2, y2)) = self.selectedLine
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            self.selectedLine = ((x1, y1), (x2, y2))

            if self.onSetPlane:
                self.onSetPlane(self)

        def onMouseMove(event):
            if selected_axis is not self.ax:
                return

            if not self.pressed:
                return
            self.selectedLine = (self.selectedLine[0], (event.xdata, event.ydata))
            if self.patch:
                self.patch.remove()
                self.patch = None

            x, y = ((self.selectedLine[0][0], self.selectedLine[1][0]), (self.selectedLine[0][1], self.selectedLine[1][1]))
            self.patch = mlines.Line2D(x, y, lw=2, color='red', alpha=1)
            self.ax.add_line(self.patch)
            self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('button_press_event', onButtonPress)
        self.fig.canvas.mpl_connect('motion_notify_event', onMouseMove)
        self.fig.canvas.mpl_connect('button_release_event', onButtonRelease)


def ComputeLineAngle(plot: PlotPlaneSelect):
    ((x1, y1), (x2, y2)) = plot.selectedLine
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    quadrant = -180 if y1 < y2 else 0

    a = abs(plot.selectedLine[0][0] - plot.selectedLine[1][0])
    b = abs(plot.selectedLine[0][1] - plot.selectedLine[1][1])
    rad = math.atan(a/b)
    angle = abs(math.degrees(rad) + quadrant)
    print(f'atan(x) :{rad}, deg: {angle}, q: {quadrant}')
    return angle


parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store', default="Gomez_T1/a001/image.mhd",  help="Input files selection regex.")
parser.add_argument('--output', action='store', default="output/",  help="Input files selection regex.")
args = parser.parse_args()


for f in glob.glob(args.input):
    fig, (HLAax, VLAax, SAax) = plt.subplots(1, 3)

    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    SA_AXIS = None

    def enter_axes(event):
        global selected_axis
        selected_axis = event.inaxes

    fig.canvas.mpl_connect('axes_enter_event', enter_axes)
    target_dir = f'{args.output}/{"/".join(f.split("/")[:-1])}'
    os.makedirs(target_dir, exist_ok=True)

    plotHLA, plotVLA, plotSA = None, None, None

    def onHLASelected(plot: PlotPlaneSelect):
        plotVLA.image.rotation3d(0,  180-ComputeLineAngle(plot), 0)
        plotVLA.redraw()

    def onVLASelected(plot: PlotPlaneSelect):
        global SA_AXIS
        ((x0, y0), (x1, y1)) = plotHLA.selectedLine
        ((x2, y2), (x3, y3)) = plotVLA.selectedLine
        res = plotHLA.image.resolution()[0]
        mapper = interp1d([0, res], [0, 1])
        mapper_rev = interp1d([0, 1], [0, plotHLA.image.resolution()[0]])
        mapper100_rev = interp1d([0, 1], [0, plotHLA.image.resolution()[2]])
        x0, y0, x1, y1, x2, y2, x3, y3 = mapper([x0, y0, x1, y1, x2, y2, x3, y3])

        VLA_line = Line(Point(x2, y2), Point(x3, y3))
        (a, b, c) = VLA_line.coefficients
        y = Symbol('y')
        z0 = 1 - float(solve(a*(1-y0) + b*y + c, y)[0])
        z1 = 1 - float(solve(a*(1-y1) + b*y + c, y)[0])
        print()
        print("Before:", (x0, y0, z0), (x1, y1, z1))
        print("Before coords:", mapper_rev((x0, y0)), mapper100_rev(z0), mapper_rev((x1, y1)), mapper100_rev(z1))
        x0, y0, z0, x1, y1, z1 = x0 - 0.5, y0 - 0.5, z0 - 0.5, x1 - 0.5, y1 - 0.5, z1 - 0.5
        print("Sub:", (x0, y0, z0), (x1, y1, z1))
        r = R.from_euler('xyz', [0, 90, 270], degrees=True)
        x0, y0, z0 = r.apply(np.array([x0, y0, z0]))
        x1, y1, z1 = r.apply(np.array([x1, y1, z1]))
        print("Sub2:", (x0, y0, z0), (x1, y1, z1))
        x0, y0, z0, x1, y1, z1 = x0 + 0.5, y0 + 0.5, z0 + 0.5, x1 + 0.5, y1 + 0.5, z1 + 0.5
        print("After:", (x0, y0, z0), (x1, y1, z1))

        x0, y0, x1, y1 = mapper_rev([x0, y0, x1, y1])
        z0, z1 = mapper100_rev([z0, z1])

        print("After coords:", (x0, y0, z0), (x1, y1, z1))

        zeroPoint = Point3D(0, 0, 0)
        XAxis = Line3D(zeroPoint, Point3D(1, 0, 0))
        YAxis = Line3D(zeroPoint, Point3D(0, 1, 0))
        ZAxis = Line3D(zeroPoint, Point3D(0, 0, 1))
        SA_AXIS = Line3D((x0, y0, z0), (x1, y1, z1))
        X_ANGLE = math.degrees(float(SA_AXIS.angle_between(XAxis)))
        Y_ANGLE = math.degrees(float(SA_AXIS.angle_between(YAxis)))
        Z_ANGLE = math.degrees(float(SA_AXIS.angle_between(ZAxis)))
        print(X_ANGLE, Y_ANGLE, Z_ANGLE)
        plotSA.image.rotation3d(0, Y_ANGLE,  -(180 - X_ANGLE))
        plotSA.redraw()

    imageHLA = ItkImage(f)
    imageHLA.rotation3d(0, 90, 270)

    plotHLA = PlotPlaneSelect(imageHLA, HLAax, onSetPlane=onHLASelected)
    plotVLA = PlotPlaneSelect(ItkImage(f), VLAax, onSetPlane=onVLASelected)
    plotSA = PlotPlaneSelect(ItkImage(f), SAax)

    def nextFile(event):
        global SA_AXIS
        if not SA_AXIS:
            print("SA AXIS not set. DO IT!")
            return
        with open(f"{target_dir}/meta.json", 'w') as fp:
            data = {
                "SA": {
                    "A": {
                        "x": float(SA_AXIS.points[0].x),
                        "y": float(SA_AXIS.points[0].y),
                        "z": float(SA_AXIS.points[0].z)
                    },
                    "B": {
                        "x": float(SA_AXIS.points[1].x),
                        "y": float(SA_AXIS.points[1].y),
                        "z": float(SA_AXIS.points[1].z),
                    },
                },
            }
            print(data)
            json.dump(data, fp)
            img = sitk.GetImageFromArray(plotSA.image.ct_scan)
            img.SetOrigin(plotSA.image.origin)
            img.SetSpacing(plotSA.image.spacing)

            sitk.WriteImage(img, target_dir + '/image.mhd')
            plt.close()

    bnext = Button(axnext, 'Save and next')
    bnext.on_clicked(nextFile)

    plt.show()
