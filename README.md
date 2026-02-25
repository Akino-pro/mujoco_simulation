MAC and MAC2 are for Mac envirnment simulation
mujoco_gen3robotiq.py is for Windows environment simulation

python mujoco_demo_gen3robotiq_infer_stub.py --transport http --endpoint https://unmetalised-jolanda-perthitic.ngrok-free.dev/infer --prompt "pick and place the large teddy bear in box" --timeout-s 120 --action-mode libero_ee_delta --midway-steps 8   