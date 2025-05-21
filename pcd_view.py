import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import copy

# Tree center coordinates with json file. Need to output with setup !!!!!!!!!!!!!!!
tree_centers = {
    368:  {"x": 1.79,  "y": -1.83}
}

def crop_pcd_by_tree(tree_id, radius, z_min, z_max, pcd):
    if tree_id not in tree_centers:
        print(f"Tree ID {tree_id} not found.")
        return None

    cx, cy = tree_centers[tree_id]["x"], tree_centers[tree_id]["y"]
    points = np.asarray(pcd.points)

    dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    mask = (dists <= radius) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    cropped_points = points[mask]

    if len(cropped_points) == 0:
        print("No points found in the specified region.")
        return None

    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)


    if pcd.has_colors():
        cropped_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

    return cropped_pcd

def fit_ground_plane_with_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                              ransac_n=ransac_n,
                                              num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    print(f"平面方程式: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"法向量: {normal}")
    return plane_model, normal, inlier_cloud, outlier_cloud, a, b, c, d

def transform_point_cloud_to_flat_ground(pcd, plane_model, normal):
    a, b, c, d = plane_model
    point_on_plane = -d * np.array([a, b, c]) / (a**2 + b**2 + c**2)  # 平面上一點

    translation = -point_on_plane
    target_normal = np.array([0, 0, 1])
    v = np.cross(normal, target_normal)
    s = np.linalg.norm(v)
    if s == 0:
        R = np.eye(3)
    else:
        c_dot = np.dot(normal, target_normal)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_dot) / (s ** 2))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(T)
    return pcd_copy, T

def print_plane_angle(normal):
    angle = np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0))
    angle_deg = np.degrees(angle)
    print(f"angle: {angle_deg:.2f}°")

# input pcd
input_pcd_path = "VAT-0723M-05.pcd"
pcd = o3d.io.read_point_cloud(input_pcd_path)

tree_id = 368
radius = 1.45
z_min = -6
z_max = 4


# crop initially
cropped = crop_pcd_by_tree(tree_id, radius, z_min, z_max, pcd)
o3d.visualization.draw_geometries([cropped])


def crop_by_z(pcd, z_min, z_max):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-np.inf, -np.inf, z_min),
                                                max_bound=(np.inf, np.inf, z_max))
    cropped = pcd.crop(bbox)
    return cropped

ground_candidate = crop_by_z(cropped, z_min=-6, z_max=-1)
o3d.visualization.draw_geometries([ground_candidate], window_name='only reserve ground point')
plane_model, normal, inlier_cloud, outlier_cloud ,a ,b ,c ,d = fit_ground_plane_with_ransac(ground_candidate,
                                                                                 distance_threshold=0.01,
                                                                                 ransac_n=3,
                                                                                 num_iterations=1000)

print_plane_angle(normal)
o3d.visualization.draw_geometries([inlier_cloud.paint_uniform_color([1,0,0]),  # 地面 inlier
                                   outlier_cloud.paint_uniform_color([0,0,1])], window_name='fit result')

cropped_aligned, T = transform_point_cloud_to_flat_ground(cropped, plane_model, normal)
o3d.visualization.draw_geometries([cropped_aligned], window_name='transferred pcd')

def select_best_cluster(slice_pts, eps=0.05, min_samples=10):
    coords = slice_pts[:, :2]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    best_score = np.inf
    best_pts, best_center, best_r = None, (0, 0), 0

    for lab in set(labels):
        if lab < 0:
            continue
        pts_lab = slice_pts[labels == lab]
        if len(pts_lab) < min_samples:
            continue
        xs, ys = pts_lab[:, 0], pts_lab[:, 1]
        A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
        b = xs ** 2 + ys ** 2
        a, b_coef, c = np.linalg.lstsq(A, b, rcond=None)[0]
        center = (a, b_coef)
        r = np.sqrt(c + a ** 2 + b_coef ** 2)

        d = np.hypot(xs - center[0], ys - center[1])
        resid = np.mean(np.abs(d - r))
        score = resid / len(pts_lab)
        if score < best_score:
            best_score = score
            best_pts = pts_lab
            best_center = center
            best_r = r
    return best_pts, best_center, best_r

def expand_circle_selection(slice_pts, center, r, margin=0.02):
    xs, ys = slice_pts[:, 0], slice_pts[:, 1]
    d = np.hypot(xs - center[0], ys - center[1])
    mask = np.abs(d - r) <= margin
    return slice_pts[mask]



def visualize_each_slice(pcd, cx, cy, z_min=1.15, z_max=1.45, dz=0.01, eps=0.01, min_samples=10, margin=0.02):
    pts = np.asarray(pcd.points)
    z_centers = np.arange(z_min + dz / 2, z_max, dz)

    for i, zc in enumerate(z_centers):
        mask_z = np.abs(pts[:, 2] - zc) <= dz / 2
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        mask_xy = dists <= 1.0
        slice_pts = pts[mask_z & mask_xy]

        if len(slice_pts) < min_samples:
            continue

        best_pts, center, r = select_best_cluster(slice_pts, eps=eps, min_samples=min_samples)

        plt.figure(figsize=(5, 5))
        plt.scatter(slice_pts[:, 0], slice_pts[:, 1], c='lightgray', s=2, label='All slice pts')

        if best_pts is not None:
            dtype = slice_pts.dtype
            mask_best = np.isin(slice_pts.view([('', dtype)] * 3), best_pts.view([('', dtype)] * 3))
            not_in_best = slice_pts[~mask_best.squeeze()]
            expanded_candidates = expand_circle_selection(not_in_best, center, r, margin=margin)
            final_pts = np.vstack([best_pts, expanded_candidates])

            # visualization
            plt.scatter(final_pts[:, 0], final_pts[:, 1], c='blue', s=4, label='Final selection')
            circle = plt.Circle(center, r, color='red', fill=False, lw=2, label='Fitted circle')
            plt.gca().add_patch(circle)

        diameter_cm = 2 * r * 100
        plt.title(f"Slice at z = {zc:.2f} m\nFitted diameter = {diameter_cm:.2f} cm")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.legend()
        plt.show()


cx, cy = tree_centers[tree_id]["x"], tree_centers[tree_id]["y"]
z_ground = -(a * cx + b * cy + d) / c # ax + by + cz + d = 0
print((z_ground))
tree_center_pt = np.array([cx, cy, 0, 1])  # homogeneous
tree_center_pt_new = T @ tree_center_pt
cx_new, cy_new = tree_center_pt_new[:2]

visualize_each_slice(
    cropped_aligned, cx_new, cy_new,
    z_min=1.1, z_max=1.5, dz=0.02,
    eps=0.05, min_samples=10,
    margin=0.05
)

# collect all layer
def collect_selected_slice_points(pcd, cx, cy, z_min=1.1, z_max=1.5, dz=0.02, eps=0.05, min_samples=10, margin=0.02):
    pts = np.asarray(pcd.points)
    z_centers = np.arange(z_min + dz / 2, z_max, dz)
    collected_pts = []

    for zc in z_centers:
        mask_z = np.abs(pts[:, 2] - zc) <= dz / 2
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        mask_xy = dists <= 1.0
        slice_pts = pts[mask_z & mask_xy]

        if len(slice_pts) < min_samples:
            continue

        best_pts, center, r = select_best_cluster(slice_pts, eps=eps, min_samples=min_samples)
        if best_pts is None:
            continue

        dtype = slice_pts.dtype
        mask_best = np.isin(slice_pts.view([('', dtype)] * 3), best_pts.view([('', dtype)] * 3))
        not_in_best = slice_pts[~mask_best.squeeze()]
        expanded_candidates = expand_circle_selection(not_in_best, center, r, margin=margin)
        final_pts = np.vstack([best_pts, expanded_candidates])
        collected_pts.append(final_pts)

    if collected_pts:
        all_pts = np.vstack(collected_pts)
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(all_pts)
        return final_pcd
    else:
        return None

combined_pcd = collect_selected_slice_points(
    cropped_aligned, cx_new, cy_new,
    z_min=1.1, z_max=1.5, dz=0.02,
    eps=0.05, min_samples=10,
    margin=0.05
)
o3d.visualization.draw_geometries([combined_pcd])

if combined_pcd:
    o3d.io.write_point_cloud(f"tree_{tree_id}_main_trunk.pcd", combined_pcd)
    print("Saved：tree_{}_main_trunk.pcd".format(tree_id))
else:
    print("Fail to find main trunk.")
