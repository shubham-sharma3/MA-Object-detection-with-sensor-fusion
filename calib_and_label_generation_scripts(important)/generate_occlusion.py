import math
import numpy as np
import os

def get_points_in_box(points, gt_box):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        cx, cy, cz = gt_box[0], gt_box[1], gt_box[2]
        dx, dy, dz, rz = gt_box[3], gt_box[4], gt_box[5], gt_box[6]
        shift_x, shift_y, shift_z = x - cx, y - cy, z - cz

        MARGIN = 1e-1
        cosa, sina = math.cos(-rz), math.sin(-rz)
        local_x = shift_x * cosa + shift_y * (-sina)
        local_y = shift_x * sina + shift_y * cosa

        mask = np.logical_and(abs(shift_z) <= dz / 2.0,
                            np.logical_and(abs(local_x) <= dx / 2.0 + MARGIN,
                                            abs(local_y) <= dy / 2.0 + MARGIN))

        points = points[mask]

        return points, mask