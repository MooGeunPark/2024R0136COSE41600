import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# PCD 파일이 있는 폴더 경로 설정
folder_path = "/Users/moogeunpark/Desktop/college/code/자율주행/data/07_straight_walk/pcd"

# 폴더에서 모든 PCD 파일 목록 가져오기
pcd_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')])

# 시각화 함수
def visualize_multiple_files_with_start_point(pcd_files, folder_path, voxel_size, min_points_in_cluster,
                                              max_points_in_cluster, min_z_value, max_z_value, 
                                              min_height, max_height, max_distance, min_x_diff, max_x_diff,
                                              min_y_diff, max_y_diff):
    # 시작 지점 설정
    start_index = int(input(f"Enter the starting index (0 to {len(pcd_files) - 1}): ").strip())
    if not (0 <= start_index < len(pcd_files)):
        print("Invalid starting index. Exiting.")
        return
    
    # Visualizer 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD Viewer", width=1600, height=1200)
    vis.get_render_option().point_size = 2.0

    # 포인트 클라우드와 바운딩 박스를 관리하기 위한 객체 초기화
    pcd_geometry = o3d.geometry.PointCloud()
    bbox_geometries = []

    idx = start_index
    while True:
        selected_file = pcd_files[idx]
        print(f"Currently visualizing: {selected_file} ({idx + 1}/{len(pcd_files)})")
        file_path = os.path.join(folder_path, selected_file)
        
        # PCD 처리 및 전처리
        original_pcd = o3d.io.read_point_cloud(file_path)
        downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)
        cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
        ror_pcd = downsample_pcd.select_by_index(ind)
        plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)
        final_point = ror_pcd.select_by_index(inliers, invert=True)
        
        # DBSCAN
        labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))
        
        # 노이즈를 제거하고 각 클러스터에 색상 지정
        max_label = labels.max()
        colors = np.zeros((len(labels), 3))
        colors[labels >= 0] = [0, 0, 1]
        
        #final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # 라벨이 0 이상인 점만 선택
        filtered_indices = np.where(labels >= 0)[0]
        filtered_pcd = final_point.select_by_index(filtered_indices)

        # 필터링된 점군에 색상 지정
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[filtered_indices])
        
        #조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
        bboxes = []
        for i in range(labels.max() + 1):
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
                                bboxes.append(bbox)

                                # 출력 부분
                                print(f"Cluster {i}:")
                                print(f"  - Number of points: {len(cluster_indices)}")
                                print(f"  - X Diff: {x_diff}")
                                print(f"  - Y Diff: {y_diff}")
                                print(f"  - Z Min: {z_min}, Z Max: {z_max}")
                                print(f"  - Z Height Difference: {height_diff}")
                                print(f"  - Max Distance from Origin: {distances.max()}")
                                print("-" * 40)
        
        # 기존 바운딩 박스를 시각화에서 제거
        for bbox in bbox_geometries:
            vis.remove_geometry(bbox)
        
        # 새롭게 계산된 PCD 및 바운딩 박스 추가
        pcd_geometry.points = filtered_pcd.points
        pcd_geometry.colors = filtered_pcd.colors
        vis.add_geometry(pcd_geometry, reset_bounding_box=False)

        bbox_geometries = bboxes  # 새 바운딩 박스 갱신
        for bbox in bboxes:
            vis.add_geometry(bbox, reset_bounding_box=False)

        # 뷰 제어 객체 가져오기
        if idx < 150:
            ctr = vis.get_view_control()
            ctr.set_zoom(0.15)
            ctr.set_lookat([0,40,0])
            ctr.set_front([0, -2, 1])
        elif 150 <= idx < 300:
            ctr = vis.get_view_control()
            ctr.set_zoom(0.15)
            ctr.set_lookat([0,30,0])
            ctr.set_front([0, -2, 1])
        else:
            ctr = vis.get_view_control()
            ctr.set_zoom(0.15)
            ctr.set_lookat([0,10,0])
            ctr.set_front([0, -2, 1])
                          
        # 화면 갱신
        vis.poll_events()
        vis.update_renderer()

        # 다음 파일로 자동 진행
        idx = (idx + 1) % len(pcd_files)
    

# 파라미터 설정
voxel_size = 0.2

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

# 실행
visualize_multiple_files_with_start_point(pcd_files, folder_path, voxel_size, min_points_in_cluster,
                                          max_points_in_cluster, min_z_value, max_z_value,
                                          min_height, max_height, max_distance, min_x_diff, max_x_diff,
                                          min_y_diff, max_y_diff)
