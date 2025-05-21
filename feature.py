import open3d as o3d
import numpy as np
import cv2
from scipy.optimize import minimize
import os
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import re


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group())

def cylinder_error(params, points, normals, w_normal=0.3):
    cx, cy, r = params
    px, py = points[:, 0], points[:, 1]
    radial_dist = np.sqrt((px - cx)**2 + (py - cy)**2)
    dist_error = (radial_dist - r)**2
    normal_z = normals[:, 2]
    normal_error = (1 - normal_z)**2
    return np.mean(dist_error + w_normal * normal_error)

# ---------- circle fit func ----------
def fit_circle(x, y):
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    a, b_coef, c = np.linalg.lstsq(A, b, rcond=None)[0]
    center = (a, b_coef)
    radius = np.sqrt(c + a ** 2 + b_coef ** 2)
    return center, radius


def extract_features(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    tree_id = int(re.findall(r"tree_(\d+)_main_trunk\.pcd", file_path)[0])

    # initial cut
    slice_mask = (points[:, 2] >= slice_z_min) & (points[:, 2] <= slice_z_max)
    slice_points = points[slice_mask]
    xy = slice_points[:, :2]
    slice_ratio = len(xy) / len(points)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    scaled = np.int32((xy - xy.min(axis=0)) * 1000)  ## unit trans

    # ellipse
    ellipse = cv2.fitEllipse(scaled)
    (cx, cy), (major, minor), angle = ellipse
    if major < minor:
        major, minor = minor, major

    dbh_ellipse = major / 1000 * 100  # cm
    eccentricity = np.sqrt(1 - (minor / major) ** 2)
    ellipse_area = np.pi * (major / 2000) * (minor / 2000)
    density = len(xy) / ellipse_area if ellipse_area > 0 else 0

    center = np.mean(xy, axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    radius_range = np.max(radii) - np.min(radii)
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)


    ellipse_mask = np.zeros((scaled[:, 1].max()+10, scaled[:, 0].max()+10), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 255, 1)
    ellipse_pts = np.column_stack(np.where(ellipse_mask > 0))
    ellipse_pts = ellipse_pts[:, [1, 0]] / 1000 + xy.min(axis=0)
    rmse = np.sqrt(np.min(cdist(xy, ellipse_pts), axis=1).mean())

    pca = PCA(n_components=2)
    pca.fit(xy)
    pca_ratio_1 = pca.explained_variance_ratio_[0]


    # cyclinder
    z_center = 1.37
    dz = 0.02
    slice_mask = (points[:, 2] >= z_center - dz / 2) & (points[:, 2] <= z_center + dz / 2)
    slice_pts = points[slice_mask]
    xs, ys = slice_pts[:, 0], slice_pts[:, 1]
    (center_x, center_y), init_r = fit_circle(xs, ys)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    normals = np.asarray(pcd.normals)
    init_params = [center_x, center_y, init_r]
    res = minimize(cylinder_error, init_params, args=(points, normals), method='Powell')
    cx, cy, r_cyl = res.x
    dbh_cylinder = r_cyl * 2 * 100  # m → cm
    print(f"Tree{tree_id} DBH by clinder ：{dbh_cylinder:.2f} cm")
    fit_rmse = np.sqrt(res.fun)
    ref_cx, ref_cy = np.mean(xy, axis=0)
    cylinder_center_offset = np.linalg.norm([cx - ref_cx, cy - ref_cy])

    # angle
    unit_normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    angle_matrix = np.arccos(np.clip(np.dot(unit_normals, unit_normals.T), -1.0, 1.0))
    angle_variance = np.var(angle_matrix)



    return {
        "filename": os.path.basename(file_path),
        "slice_ratio": round(slice_ratio, 3),
        "density": round(density, 3),
        "std_radius": round(std_radius, 3),
        "radius_range": round(radius_range, 3),
        "mean_radius": round(mean_radius, 3),
        "pca_ratio_1": round(pca_ratio_1, 4), # PCA
        "dbh_ellipse_cm": round(dbh_ellipse, 2), # elipse
        "dbh_cylinder_cm": round(dbh_cylinder, 2), # cylinder
        "flatness": round(major / minor, 3), # ellipse
        "eccentricity": round(eccentricity, 4), # elipse
        "ellipse_fit_rmse": round(rmse, 4), # elipse
        "normal_angle_variance": round(angle_variance, 4),
        "cylinder_fit_rmse": round(fit_rmse, 4),
        "cylinder_center_offset": round(cylinder_center_offset, 4)
    }

# ---------- set up ----------
input_folder = "pcd"
output_excel = "features.xlsx"
slice_z_min, slice_z_max = 1.34, 1.40
files = sorted(os.listdir(input_folder), key=extract_number)
# ---------- main func ----------
features = []
for file in files:
    if file.endswith(".pcd"):
        path = os.path.join(input_folder, file)
        feature = extract_features(path)
        if feature:
            features.append(feature)

df = pd.DataFrame(features)
df.to_excel(output_excel, index=False)
print(f"Saved: {output_excel}")
