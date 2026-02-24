import socket
import struct

import numpy as np
import cv2  # here we DO use imshow safely (separate process)

HOST = "127.0.0.1"
PORT = 5005

def recv_exact(conn, nbytes: int) -> bytes:
    buf = b""
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[INFO] Wrist receiver listening on {HOST}:{PORT}")
    conn, addr = srv.accept()
    print(f"[INFO] Connected by {addr}")

    cv2.namedWindow("Wrist Camera (TCP)", cv2.WINDOW_NORMAL)

    try:
        while True:
            header = recv_exact(conn, 4)
            if not header:
                break
            (n,) = struct.unpack("!I", header)
            payload = recv_exact(conn, n)
            if not payload:
                break

            arr = np.frombuffer(payload, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            cv2.imshow("Wrist Camera (TCP)", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            srv.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()