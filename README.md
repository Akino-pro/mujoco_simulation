
Yuxing:
MAC and MAC2 are for Mac envirnment simulation
mujoco_gen3robotiq.py is for Windows environment simulation

python mujoco_demo_gen3robotiq_infer_stub.py --transport http --endpoint https://unmetalised-jolanda-perthitic.ngrok-free.dev/infer --prompt "pick and place the large teddy bear in box" --timeout-s 120 --action-mode libero_ee_delta --midway-steps 8   


zhiang:
第一步，启动一个后台接收器，用于实时显示机械臂末端相机传回的画面。python MAC2.py 请保持此终端在后台运行，不要关闭。（作用： 开启一个 Socket 服务端并弹出一个窗口，实时显示当前的 RGB 彩色画面和深度图（Depth Map））

第二步：启动物理仿真与环绕扫描打开一个新的终端（确保已激活虚拟环境），启动 MuJoCo 物理引擎。macOS 用户特别注意：必须使用 mjpython 命令启动，否则会导致系统图形界面主线程冲突崩溃 指令：“mjpython MAC.py” 
Linux/Windows 用户请使用 “python MAC.py” 命令
作用： 
1. 加载机械臂和温室（西红柿）的 3D 物理模型。
2. 控制机械臂执行预设的“半月形环绕轨迹”。
3. 沿途不断拍摄，将局部相机坐标系下的点云通过矩阵乘法转换为绝对世界坐标并不断累加。
4. 实时计算并记录每一帧的相机内参 K 和 4x4 外参位姿。结果： 机械臂走完轨迹（或按 Ctrl+C 退出）后，会在本地生成中间数据文件 tomato_scan.npz 和相机轨迹记录 camera_trajectory.json。

第三步：点云可视化与数据集导出仿真结束后，运行可视化脚本，将生成的中间数据转换为下游 AI 算法通用的标准格式。指令：“python view_3d.py”
作用： 读取生成的 .npz 文件，通过 Open3D 渲染出拼接好的全局 3D 点云，并自动导出 .csv 和 .ply 格式的最终文件。结果是会弹出一个 3D 交互窗口（带有世界坐标轴），供你鼠标拖拽查看纯净的西红柿点云图。
