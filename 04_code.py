import open3d as o3d
import numpy as np
import os
import cv2  # OpenCV를 사용하여 동영상 생성

# PCD 파일이 있는 폴더 경로 설정
folder_path = "/Users/moogeunpark/Desktop/college/code/자율주행/data/04_zigzag_walk/pcd"

# 폴더에서 모든 PCD 파일 목록 가져오기
pcd_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')])

def visualize_multiple_files_with_start_point(pcd_files, folder_path, voxel_size, min_points_in_cluster,
                                              max_points_in_cluster, min_z_value, max_z_value, 
                                              min_height, max_height, max_distance, output_video_path, max_frames=100):
    # Visualizer 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD Viewer", width=1600, height=1200)
    vis.get_render_option().point_size = 2.0

    # 포인트 클라우드와 바운딩 박스를 관리하기 위한 객체 초기화
    pcd_geometry = o3d.geometry.PointCloud()
    bbox_geometries = []

    idx = 0
    frame_images = []  # 프레임 이미지 리스트
    min_x_diff = 0.4  # x 차이 최소값
    max_x_diff = 1.0  # x 차이 최대값
    min_y_diff = 0.4  # y 차이 최소값
    max_y_diff = 1.0  # y 차이 최대값

    # OpenCV를 사용하여 동영상 생성 준비
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 동영상 인코딩 형식 설정 (mp4v)
    
    # 첫 번째 이미지 캡처 후 동영상 크기 설정
    selected_file = pcd_files[idx]
    file_path = os.path.join(folder_path, selected_file)
    original_pcd = o3d.io.read_point_cloud(file_path)
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    image = vis.capture_screen_float_buffer(True)
    image = np.asarray(image)  # float buffer를 NumPy 배열로 변환
    frame_height, frame_width, _ = image.shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

    while idx < len(pcd_files) and len(frame_images) < max_frames:  # 최대 프레임수 제한
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
        colors = np.zeros((len(labels), 3))
        colors[labels >= 0] = [0, 0, 1]
        final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        # 필터링 및 바운딩 박스
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
        
        # 기존 바운딩 박스를 시각화에서 제거
        for bbox in bbox_geometries:
            vis.remove_geometry(bbox)
        
        # 새롭게 계산된 PCD 및 바운딩 박스 추가
        pcd_geometry.points = final_point.points
        pcd_geometry.colors = final_point.colors
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
        else:
            ctr = vis.get_view_control()
            ctr.set_zoom(0.15)
            ctr.set_lookat([0,10,0])
            ctr.set_front([0, -2, 1])
                          
        # 화면 갱신
        vis.poll_events()
        vis.update_renderer()

        # 이미지 캡처 및 저장
        image = vis.capture_screen_float_buffer(True)
        image = np.asarray(image)  # float buffer를 NumPy 배열로 변환

        # 0~255 범위로 변환하고 uint8로 캐스팅
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
        frame_bgr = np.clip(frame_bgr * 255, 0, 255).astype(np.uint8)  # [0, 1] 범위를 [0, 255]로 변경 후 uint8로 변환

        video_writer.write(frame_bgr)  # 동영상에 프레임 추가

        # 다음 파일로 자동 진행
        idx += 1

    # 동영상 파일 저장
    video_writer.release()
    print(f"Video saved at {output_video_path}")


# 파라미터 설정
voxel_size = 0.2
min_points_in_cluster = 50
max_points_in_cluster = 80
min_z_value = -3.0
max_z_value = 3.0
min_height = 1.0
max_height = 2.0
max_distance = 30.0
output_video_path = '/Users/moogeunpark/Desktop/college/code/자율주행/data/output_video.mp4'  # 동영상 파일 경로

# 실행
visualize_multiple_files_with_start_point(pcd_files, folder_path, voxel_size, min_points_in_cluster,
                                          max_points_in_cluster, min_z_value, max_z_value,
                                          min_height, max_height, max_distance, output_video_path)
