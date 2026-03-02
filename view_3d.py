import numpy as np
import open3d as o3d
print("正在加载全局 3D 点云数据...")
data = np.load("tomato_scan.npz")
pts = data['points']
clrs = data['colors']

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(clrs)

o3d.io.write_point_cloud("tomato_global_pointcloud.ply", pcd)
print(f"✅标准点云图已保存: tomato_global_pointcloud.ply")

combined_data = np.hstack((pts, clrs))
np.savetxt("tomato_global_coordinates.csv", combined_data, delimiter=",",
           header="X,Y,Z,R,G,B", comments="", fmt='%.5f')
print(f"✅纯坐标数据已保存: tomato_global_coordinates.csv")

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

print("弹窗显示中... (请查看加入真实世界坐标系后的点云)")
o3d.visualization.draw_geometries([pcd, mesh_frame], window_name="Global 3D Tomato")