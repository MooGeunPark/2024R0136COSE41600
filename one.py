# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import random

file_path = r"/Users/moogeunpark/Desktop/college/code/자율주행/data/07_straight_walk/pcd/pcd_000365.pcd"

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = ror_pcd.select_by_index(inliers, invert=True)

# 포인트 클라우드를 NumPy 배열로 변환
points = np.asarray(final_point.points)

# DBSCAN 클러스터링 적용
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))

# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = np.zeros((len(labels), 3))
colors[labels >= 0] = [0, 0, 1]
#final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

filtered_indices = np.where(labels >= 0)[0]
filtered_pcd = final_point.select_by_index(filtered_indices)

# 필터링된 점군에 색상 지정
filtered_pcd.colors = o3d.utility.Vector3dVector(colors[filtered_indices])

min_points_in_cluster = 45
max_points_in_cluster = 100
min_z_value = -1.5
max_z_value = 1.5
min_height = 1.0
max_height = 1.7
min_x_diff = 0.4  # x 차이 최소값
max_x_diff = 1.0  # x 차이 최대값
min_y_diff = 0.4  # y 차이 최소값
max_y_diff = 1.0  # y 차이 최대값
max_distance = 30

# 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
bboxes_1234 = []
for i in range(max_label + 1):
    cluster_indices = np.where(labels == i)[0]
    if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
        cluster_pcd = final_point.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        
        # x, y, z 값 추출
        x_values = points[:, 0]
        y_values = points[:, 1]
        z_values = points[:, 2]
        
        # 각 축의 최대, 최소값 차이 계산
        x_diff = x_values.max() - x_values.min()
        y_diff = y_values.max() - y_values.min()
        z_min = z_values.min()
        z_max = z_values.max()

        # 조건 추가: x, y 차이와 기존 조건
        if min_x_diff <= x_diff <= max_x_diff and min_y_diff <= y_diff <= max_y_diff:
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= max_distance:
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (1, 0, 0)
                        bboxes_1234.append(bbox)

                        # 출력 부분
                        print(f"Cluster {i}:")
                        print(f"  - Number of points: {len(cluster_indices)}")
                        print(f"  - X Diff: {x_diff}")
                        print(f"  - Y Diff: {y_diff}")
                        print(f"  - Z Min: {z_min}, Z Max: {z_max}")
                        print(f"  - Z Height Difference: {height_diff}")
                        print(f"  - Max Distance from Origin: {distances.max()}")
                        print("-" * 40)

# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1600, height=1200)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size

    # 뷰 제어 객체 가져오기
    ctr = vis.get_view_control()

    ctr = vis.get_view_control()
    ctr.set_zoom(0.15)
    ctr.set_lookat([0,10,0])
    ctr.set_front([0, -2, 1])

    vis.run()
    vis.destroy_window()

# 시각화 (포인트 크기를 원하는 크기로 조절 가능)
visualize_with_bounding_boxes(filtered_pcd, bboxes_1234, point_size=2.0)
