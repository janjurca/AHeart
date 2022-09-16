import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector


class ItkImage:
    def __init__(self, filename: str) -> None:
        self.image = sitk.ReadImage(filename, imageIO="MetaImageIO")
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        self.ct_scan = sitk.GetArrayFromImage(self.image)
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        self.origin = np.array(list(reversed(self.image.GetOrigin())))
        # Read the spacing along each dimension
        self.spacing = np.array(list(reversed(self.image.GetSpacing())))

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
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)
        euler_transform = sitk.Euler3DTransform(self.get_center(), theta_x, theta_y, theta_z, (0, 0, 0))
        euler_transform.SetCenter(self.get_center())
        euler_transform.SetRotation(theta_x, theta_y, theta_z)
        resampled_image = self.resample(euler_transform)

        return resampled_image


class Plot:
    def __init__(self, image: ItkImage) -> None:
        self.image = image
        self.fig, self.ax = plt.subplots()
        self.index = int(len(self.image.ct_scan)/2)
        self.boundingbox = None
        self.patch = None

        def onScroll(event):
            if event.button == "up":
                self.index += 1
            if event.button == "down":
                self.index -= 1
            self.index = 0 if self.index < 0 else (len(self.image.ct_scan) - 1 if self.index > len(self.image.ct_scan) else self.index)
            self.ax.imshow(self.image.ct_scan[self.index])
            self.ax.set_title(f"Slice: {self.index}")
            self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('scroll_event', onScroll)

    def show(self) -> None:
        self.ax.imshow(self.image.ct_scan[self.index])

        def areaSelect(click, release):
            self.boundingbox = ((click.xdata, click.ydata), (release.xdata, release.ydata))
            if self.patch:
                self.patch.remove()
            self.patch = self.ax.add_patch(
                patches.Rectangle(
                    self.boundingbox[0],
                    self.boundingbox[1][0] - self.boundingbox[0][0],
                    self.boundingbox[1][1] - self.boundingbox[0][1],
                    linewidth=1, edgecolor='r', facecolor='none'))

        rect_selector = RectangleSelector(self.ax, areaSelect, button=[1])
        plt.show()


image = ItkImage("test_data/image.mhd")
plot = Plot(image)

plot.show()
