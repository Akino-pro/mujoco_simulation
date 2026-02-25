import numpy as np
import open3d as o3d

print("正在加载 3D 点云数据...")
# 1. 读取刚才保存的 numpy 数据
data = np.load("tomato_scan.npz")
pts = data['points']
clrs = data['colors']

# 2. 扔给 Open3D 去渲染
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(clrs)

print("加载成功！使用鼠标左键拖拽旋转，滚轮缩放。")
o3d.visualization.draw_geometries([pcd], window_name="My 3D Tomato")