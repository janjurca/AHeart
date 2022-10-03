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

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
plane1_angle = 0
plane2_angle = 0

selected_axis = None


def enter_axes(event):
    global selected_axis
    selected_axis = event.inaxes


fig.canvas.mpl_connect('axes_enter_event', enter_axes)
#fig.canvas.mpl_connect('axes_leave_event', leave_axes)


class ItkImage:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.load()

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
        return self.image.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                                         int(np.ceil(height/2)),
                                                         int(np.ceil(depth/2))))

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
        euler_transform.SetCenter(self.get_center())
        euler_transform.SetRotation(theta_x, theta_y, theta_z)
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


class PlotBoundingBox(VolumeImage):
    def __init__(self, image: ItkImage, ax, onSetBoundingBoxFunc=None, title="BoundingBox") -> None:
        self.image = image
        self.fig = fig
        self.ax = ax
        self.index = int(len(self.image.ct_scan)/2)
        self.boundingbox = None
        self.patch = None
        self.onSetBoundingBoxFunc = onSetBoundingBoxFunc
        self.ax_data = self.ax.imshow(self.image.ct_scan[self.index])
        self.eventSetup()

        def areaSelect(click, release):
            self.boundingbox = ((click.xdata, click.ydata), (release.xdata, release.ydata))
            if self.patch:
                self.patch.remove()
                self.patch = None
            self.patch = self.ax.add_patch(
                patches.Rectangle(
                    self.boundingbox[0],
                    self.boundingbox[1][0] - self.boundingbox[0][0],
                    self.boundingbox[1][1] - self.boundingbox[0][1],
                    linewidth=1, edgecolor='r', facecolor='none'))
            if self.onSetBoundingBoxFunc:
                self.onSetBoundingBoxFunc(self)

        self.rect_selector = RectangleSelector(self.ax, areaSelect, button=[1])


class PlotPlaneSelect(VolumeImage):
    def __init__(self, image: ItkImage, ax, onSetPlane=None, title="PlaneSelect") -> None:
        self.image = image
        self.fig = fig
        self.ax = ax
        self.index = int(len(self.image.ct_scan)/2)
        self.plane = None
        self.patch = None
        self.onSetPlane = onSetPlane
        self.ax_data = self.ax.imshow(self.image.ct_scan[self.index])
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


parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store', default="test_data/image.mhd",  help="Input files selection regex.")
args = parser.parse_args()

for f in glob.glob(args.input):
    plane1_angle = 0
    plane2_angle = 0

    plot_ps3 = PlotPlaneSelect(ItkImage(f), ax4, title="Plane 3 select")

    def onPlane2Set(plot: PlotPlaneSelect):
        global plane1_angle
        global plane2_angle
        ((x1, y1), (x2, y2)) = plot.selectedLine
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        quadrant = -180 if y1 < y2 else 0

        a = abs(plot.selectedLine[0][0] - plot.selectedLine[1][0])
        b = abs(plot.selectedLine[0][1] - plot.selectedLine[1][1])
        rad = math.atan(a/b)
        plane2_angle = abs(math.degrees(rad) + quadrant)
        print(f'atan(x) :{rad}, deg: {plane2_angle}, q: {quadrant}')
        print(plane2_angle, plane1_angle)
        plot_ps3.image.rotation3d(0, plane1_angle, plane2_angle)
        plot_ps3.redraw()

    plot_ps2 = PlotPlaneSelect(ItkImage(f), ax3, onSetPlane=onPlane2Set, title="Plane 2 select")

    def onPlane1Set(plot: PlotPlaneSelect):
        global plane1_angle
        ((x1, y1), (x2, y2)) = plot.selectedLine
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        quadrant = -180 if y1 < y2 else 0

        a = abs(plot.selectedLine[0][0] - plot.selectedLine[1][0])
        b = abs(plot.selectedLine[0][1] - plot.selectedLine[1][1])
        rad = math.atan(a/b)
        plane1_angle = abs(math.degrees(rad) + quadrant)

        print(f'atan(x) :{rad}, deg: {plane1_angle}, q: {quadrant}')
        plot_ps2.image.rotation3d(0, plane1_angle, 0)
        plot_ps2.redraw()

    plot_ps1 = PlotPlaneSelect(ItkImage(f), ax2, title="Plane 1 select", onSetPlane=onPlane1Set)

    def onBoundingBoxSet(plot: PlotBoundingBox):
        ((x1, y1), (x2, y2)) = plot.boundingbox
        width, height, depth = plot_bb.image.resolution()
        m = interp1d([0, height], [0, depth])
        _slice = int(m(y1 + abs(y2-y1)*0.6))

        plot_ps1.image.rotation3d(180, 90, 90)
        plot_ps1.setIndex(_slice)
        plot_ps1.redraw()

    plot_bb = PlotBoundingBox(ItkImage(f), ax1, onSetBoundingBoxFunc=onBoundingBoxSet)

    def nextFile(event):
        print("Next was pushed")
        # TODO HANDLE figure close and proper reinit for other image

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(nextFile)

    plt.show()
    # TODO Handle data save
