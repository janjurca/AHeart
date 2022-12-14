from typing import Tuple
import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import matplotlib.lines as mlines
import math
import argparse
import glob
import torch
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

    def construct_transf_matrix_3D(params, resize_factor):
        
        thx, thy, thz, tx, ty, tz = params
        
        theta = torch.zeros(1, 4, 4) 
        theta[1,3,3] = 1
        
        R_z = torch.zeros(  3, 3) 
        R_z[1,2,2] = 1
        R_z[1,0,0] = torch.cos(thz )
        R_z[1,0,1] = -torch.sin(thz )
        R_z[1,1,0] = torch.sin(thz )
        R_z[1,1,1] =  torch.cos(thz )
        
        R_y = torch.zeros(  3, 3) 
        R_y[1,1,1] = 1
        R_y[1,0,0] = torch.cos(thy )
        R_y[1,2,0] = -torch.sin(thy )
        R_y[1,0,2] = torch.sin(thy )
        R_y[1,2,2] =  torch.cos(thy )
        
        R_x = torch.zeros(  3, 3) 
        R_x[1,0,0] = 1
        R_x[1,1,1] = torch.cos(thx )
        R_x[1,2,1] = -torch.sin(thx )
        R_x[1,1,2] = torch.sin(thx )
        R_x[1,2,2] =  torch.cos(thx )
        
        R = torch.bmm( torch.bmm(R_z, R_y), R_x )
        # R = torch.bmm(R_z, R_y)
        
        T = torch.zeros(  3, 3) 
        T[1,0,0] = resize_factor
        T[1,1,1] = resize_factor
        T[1,2,2] = resize_factor
    
        theta[1,0:3,0:3] = torch.bmm( T , R )
        
        theta[1,0,3] = tx 
        theta[1,1,3] = ty 
        theta[1,2,3] = tz 

        # theta[:,0,0] = resize_factor * torch.cos(angle_rad[inds])
        # theta[:,1,1] = resize_factor * torch.cos(angle_rad[inds])
        # theta[:,0,1] = torch.sin(angle_rad[inds])
        # theta[:,1,0] = -torch.sin(angle_rad[inds])
        
        # theta[:,0,2] = tx[inds]
        # theta[:,1,2] = ty[inds]
        
        return theta


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
    return angle


parser = argparse.ArgumentParser()
# parser.add_argument('--input', action='store', default="D:\Skola\Doktorske\Projects\Cardio\synth\krychla.mhd",  help="Input files selection regex.")
parser.add_argument('--input', action='store', default="E:\Projects\Cardio\pokus2\krychla_x.mhd",  help="Input files selection regex.")
# parser.add_argument('--input', action='store', default="Gomez_T1/a005/image.mhd",  help="Input files selection regex.")
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
        
        plotVLA.image.rotation3d(0,  180-ComputeLineAngle(plot), 0)
        plotVLA.redraw()

    def onVLASelected(plot: PlotPlaneSelect):
  
        global HLA_AXIS
        ((x0, y2), (x1, y3)) = plotInit.selectedLine
        ((z0, y0), (z1, y1)) = plotVLA.selectedLine
        res_Init_x = plotInit.image.resolution()[0]
        res_Init_y = plotInit.image.resolution()[1]
        res_VLA_z = plotVLA.image.resolution()[0]
        res_VLA_y = plotVLA.image.resolution()[1]
        mapper_init_x = interp1d([0, res_Init_x], [0, 1])
        mapper_init_y = interp1d([0, res_Init_y], [0, 1])
        mapper_VLA_z = interp1d([0, res_VLA_z], [0, 1])
        mapper_VLA_y = interp1d([0, res_VLA_y], [0, 1])
        z0, z1 = mapper_VLA_z([z0, z1])
        y0, y1 = mapper_VLA_y([y0, y1]) 
        x0, x1 = mapper_init_x([x0, x1])
        y2, y3 = mapper_init_y([y2, y3])


        # x0, y0, z0, x1, y1, z1 = x0 - 0.5, y0 - 0.5, z0 - 0.5, x1 - 0.5, y1 - 0.5, z1 - 0.5
        r = R.from_euler('x', 90, degrees=True)
        x0, y0, z0 = r.apply(np.array([x0, y0, z0]))
        x1, y1, z1 = r.apply(np.array([x1, y1, z1]))
        # x0, y0, z0, x1, y1, z1 = x0 + 0.5, y0 + 0.5, z0 + 0.5, x1 + 0.5, y1 + 0.5, z1 + 0.5


        # x0, y0, z0, x1, y1, z1 = x0 - 0.5, y0 - 0.5, z0 - 0.5, x1 - 0.5, y1 - 0.5, z1 - 0.5
        # r = R.from_euler('xyz', [0, 0, 90], degrees=True)
        # x0, y0, z0 = r.apply(np.array([x0, y0, z0]))
        # x1, y1, z1 = r.apply(np.array([x1, y1, z1]))
        # x0, y0, z0, x1, y1, z1 = x0 + 0.5, y0 + 0.5, z0 + 0.5, x1 + 0.5, y1 + 0.5, z1 + 0.5

        mapper_rev_init_x = interp1d([0, 1], [0, res_Init_x])
        mapper_rev_init_y = interp1d([0, 1], [0, res_Init_y])
        mapper_rev_VLA_z = interp1d([0, 1], [0, res_VLA_z])
        mapper_rev_VLA_y = interp1d([0, 1], [0, res_VLA_y])
        z0, z1 = mapper_rev_VLA_z([z0, z1])
        y0, y1 = mapper_rev_VLA_y([y0, y1]) 
        x0, x1 = mapper_rev_init_x([x0, x1])
        y2, y3 = mapper_rev_init_y([y2, y3])

        zeroPoint = Point3D(0, 0, 0)
        XAxis = Line3D(zeroPoint, Point3D(1, 0, 0))
        YAxis = Line3D(zeroPoint, Point3D(0, 1, 0))
        ZAxis = Line3D(zeroPoint, Point3D(0, 0, 1))
        HLA_AXIS = Line3D((x0, y0, z0), (x1, y1, z1))
        X_ANGLE = math.degrees(float(HLA_AXIS.angle_between(XAxis)))
        Y_ANGLE = math.degrees(float(HLA_AXIS.angle_between(YAxis)))
        Z_ANGLE = math.degrees(float(HLA_AXIS.angle_between(ZAxis)))
        plotHLA.image.rotation3d(Z_ANGLE, Y_ANGLE, X_ANGLE )
        plotHLA.redraw()

    def onHLASelected(plot: PlotPlaneSelect):

        global SA_AXIS
        ((x0, y2), (x1, y3)) = plotVLA.selectedLine
        ((z0, y0), (z1, y1)) = plotHLA.selectedLine
        res = plotHLA.image.resolution()[0]
        mapper = interp1d([0, res], [0, 1])
        mapper_rev = interp1d([0, 1], [0, plotVLA.image.resolution()[0]])
        mapper100_rev = interp1d([0, 1], [0, plotHLA.image.resolution()[2]])
        x0, y2, x1, y3, z0, y0, z1, y1 = mapper([x0, y2, x1, y3, z0, y0, z1, y1])

        x0, y0, z0, x1, y1, z1 = x0 - 0.5, y0 - 0.5, z0 - 0.5, x1 - 0.5, y1 - 0.5, z1 - 0.5
        r = R.from_euler('xyz', [0, 90, 270], degrees=True)
        x0, y0, z0 = r.apply(np.array([x0, y0, z0]))
        x1, y1, z1 = r.apply(np.array([x1, y1, z1]))
        x0, y0, z0, x1, y1, z1 = x0 + 0.5, y0 + 0.5, z0 + 0.5, x1 + 0.5, y1 + 0.5, z1 + 0.5

        x0, x1 = mapper_rev([x0, x1])
        z0, y0, z1, y1 = mapper100_rev([z0, y0, z1, y1])

        zeroPoint = Point3D(0, 0, 0)
        XAxis = Line3D(zeroPoint, Point3D(1, 0, 0))
        YAxis = Line3D(zeroPoint, Point3D(0, 1, 0))
        ZAxis = Line3D(zeroPoint, Point3D(0, 0, 1))
        SA_AXIS = Line3D((x0, y0, z0), (x1, y1, z1))
        X_ANGLE = math.degrees(float(SA_AXIS.angle_between(XAxis)))
        Y_ANGLE = math.degrees(float(SA_AXIS.angle_between(YAxis)))
        Z_ANGLE = math.degrees(float(SA_AXIS.angle_between(ZAxis)))
        plotSA.image.rotation3d(Z_ANGLE, Y_ANGLE, X_ANGLE )
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
