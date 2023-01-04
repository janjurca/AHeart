from typing import Tuple
import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import matplotlib.lines as mlines
import math
import argparse
import glob
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import os
import json
from scipy.spatial.transform import Rotation as R
from sympy import Point3D
from sympy.geometry import Line3D

fig, (InitAx, VLAax, HLAax, SAax) = None, (None, None, None, None)


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
        self.ct_scan = sitk.GetArrayFromImage(self.image)
        self.origin = np.array(list(reversed(self.image.GetOrigin())))  # TODO handle different rotations
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

        def onButtonsPressed(event):
            if selected_axis is not self.ax:
                return

            self.pressed = True
            if self.selectedLine[0]!= None and self.selectedLine[1]!= None:
                self.selectedLine = ((event.xdata, event.ydata, self.index), None)
            elif self.selectedLine[0]!= None and self.selectedLine[1]== None:
                self.selectedLine = (self.selectedLine[0], (event.xdata, event.ydata, self.index))
                
                self.pressed = False

                ((x1, y1, z1), (x2, y2, z2)) = self.selectedLine
                if x1 > x2:
                    x1, y1, z1, x2, y2, z2 = x2, y2, z2, x1, y1, z1
                self.selectedLine = ((x1, y1, z1), (x2, y2, z2))

                if self.onSetPlane:
                    self.onSetPlane(self)
            elif self.selectedLine[0]== None and self.selectedLine[1]== None:
                self.selectedLine = ((event.xdata, event.ydata, self.index), None)
            
            self.pressed = False

        def onMouseMove(event):
            if selected_axis is not self.ax:
                return

            if not self.pressed:
                return
            # self.selectedLine = (self.selectedLine[0], (event.xdata, event.ydata))
            # if self.patch:
            #     self.patch.remove()
            #     self.patch = None

            # x, y = ((self.selectedLine[0][0], self.selectedLine[1][0]), (self.selectedLine[0][1], self.selectedLine[1][1]))
            # self.patch = mlines.Line2D(x, y, lw=2, color='red', alpha=1)
            # self.ax.add_line(self.patch)
            # self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('motion_notify_event', onMouseMove)
        self.fig.canvas.mpl_connect('button_press_event', onButtonsPressed)


def ComputeLineAngles(plot: PlotPlaneSelect):
    ((x1, y1, z1), (x2, y2, z2)) = plot.selectedLine
    if x1 > x2:
        x1, y1, z1, x2, y2, z2 = x2, y2, z2, x1, y1, z1

    res_x = plot.image.resolution()[0]
    res_y = plot.image.resolution()[1]
    res_z = plot.image.resolution()[2]

    mapper_init_x = interp1d([0, res_x], [0, 1])
    mapper_init_y = interp1d([0, res_y], [0, 1])
    mapper_init_z = interp1d([0, res_z], [0, 1])
   
    x1, x2 = mapper_init_x([x1, x2])
    y1, y2 = mapper_init_y([y1, y2])
    z1, z2 = mapper_init_y([z1, z2])

    zeroPoint = Point3D(0, 0, 0)
    XAxis = Line3D(zeroPoint, Point3D(1, 0, 0))
    YAxis = Line3D(zeroPoint, Point3D(0, 1, 0))
    ZAxis = Line3D(zeroPoint, Point3D(0, 0, 1))
    Axis = Line3D((x1, y1, z1), (x2, y2, z2))
    X_ANGLE = math.degrees(float(Axis.angle_between(XAxis)))
    Y_ANGLE = math.degrees(float(Axis.angle_between(YAxis)))
    Z_ANGLE = math.degrees(float(Axis.angle_between(ZAxis)))

    return X_ANGLE, Y_ANGLE, Z_ANGLE


parser = argparse.ArgumentParser()
# parser.add_argument('--input', action='store', default="D:\Skola\Doktorske\Projects\Cardio\synth\krychla.mhd",  help="Input files selection regex.")
# parser.add_argument('--input', action='store', default="E:\Projects\Cardio\pokus2\krychla_x.mhd",  help="Input files selection regex.")

parser.add_argument('--input', action='store', default="Gomez_T1/a005/image.mhd",  help="Input files selection regex.")
parser.add_argument('--output', action='store', default="output/",  help="Input files selection regex.")
args = parser.parse_args()


for f in glob.glob(args.input):
    fig, (InitAx, VLAax, HLAax, SAax) = plt.subplots(1, 4)

    target_dir = f'{args.output}/{"/".join(f.split(os.sep)[-3:-1])}'
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    SA_AXIS = None

    def enter_axes(event):
        global selected_axis
        selected_axis = event.inaxes

    fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    plotInit, plotHLA, plotVLA, plotSA = None, None, None, None


    def onInitSelected(plot: PlotPlaneSelect):
        x_angle, y_angle, z_angle = ComputeLineAngles(plot)
        
        # plt.plot(plot.selectedLine[0], marker='v', color="red")
        # plotVLA.image.rotation3d(x_angle,  y_angle, z_angle)
        plotVLA.image.rotation3d(x_angle, z_angle, y_angle)
        plotVLA.redraw()

    def onVLASelected(plot: PlotPlaneSelect):
  
        global HLA_AXIS

        x_angle_Init, y_angle_Init, z_angle_Init = ComputeLineAngles(plotInit)
        x_angle_VLA, y_angle_VLA, z_angle_VLA = ComputeLineAngles(plotVLA)

        plotHLA.image.rotation3d(z_angle_Init, y_angle_VLA-90, x_angle_VLA)
 
        # plotHLA.image.rotation3d(z_angle_Init, x_angle_VLA, 90+y_angle_VLA )
        plotHLA.redraw()

    def onHLASelected(plot: PlotPlaneSelect):

        x_angle_HLA, y_angle_HLA, z_angle_HLA = ComputeLineAngles(plotHLA)
        x_angle_VLA, y_angle_VLA, z_angle_VLA = ComputeLineAngles(plotVLA)

        # plotSA.image.rotation3d(x_angle_VLA, y_angle_HLA, z_angle_HLA )
        plotSA.image.rotation3d(x_angle_VLA, x_angle_HLA, y_angle_HLA )
        plotSA.redraw()


    imageInit = ItkImage(f)
    imageInit.rotation3d(0, 90, 270)

    plotInit = PlotPlaneSelect(imageInit, InitAx, onSetPlane=onInitSelected)
    plotVLA = PlotPlaneSelect(ItkImage(f), VLAax, onSetPlane=onVLASelected)
    plotHLA = PlotPlaneSelect(ItkImage(f), HLAax, onSetPlane=onHLASelected)
    plotSA = PlotPlaneSelect(ItkImage(f), SAax)

    def nextFile(event):
        global SA_AXIS, target_dir

        os.makedirs(target_dir, exist_ok=True)

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
