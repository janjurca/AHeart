from fileinput import filename
from time import sleep
from typing import Tuple
import SimpleITK as sitk
import math
import numpy as np
import argparse
import json
from scipy.spatial.transform import Rotation as R
from sympy import Point3D
from sympy.geometry import Line3D


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
        reference_image = self.image
        interpolator = sitk.sitkBSpline
        default_value = 0
        return sitk.Resample(self.image, reference_image, transform,
                             interpolator, default_value)

    def get_center(self):
        width, height, depth = self.image.GetSize()
        p = self.image.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                                     int(np.ceil(height/2)),
                                                     int(np.ceil(depth/2))))
        print("center", p)
        return p

    def rotation3d(self, theta_x, theta_y, theta_z):
        self.load()
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)
        euler_transform = sitk.Euler3DTransform(self.get_center(), theta_x, theta_y, theta_z, (0, 0, 0))
        self.image = self.resample(euler_transform)
        self.refresh()


parser = argparse.ArgumentParser()
parser.add_argument('--mhd', action='store', default="Gomez_T1/a001/image.mhd",  help="Input files selection regex.")
parser.add_argument('--json', action='store', default="output/Gomez_T1/a001/meta.json",  help="json metas.")
args = parser.parse_args()


metas = None
with open(args.json) as fp:
    metas = json.load(fp)

zeroPoint = Point3D(0, 0, 0)
XAxis = Line3D(zeroPoint, Point3D(1, 0, 0))
YAxis = Line3D(zeroPoint, Point3D(0, 1, 0))
ZAxis = Line3D(zeroPoint, Point3D(0, 0, 1))

SA_AXIS = Line3D((metas["SA"]["A"]["x"], metas["SA"]["A"]["y"], metas["SA"]["A"]["y"]), (metas["SA"]["B"]["x"], metas["SA"]["B"]["y"], metas["SA"]["B"]["z"]))
X_ANGLE = math.degrees(float(SA_AXIS.angle_between(XAxis)))
Y_ANGLE = math.degrees(float(SA_AXIS.angle_between(YAxis)))
Z_ANGLE = math.degrees(float(SA_AXIS.angle_between(ZAxis)))


imageSA = ItkImage(args.mhd)
imageSA.rotation3d(0, Y_ANGLE,  -(180 - X_ANGLE))

sitk.Show(imageSA.image, title="cthead1")
