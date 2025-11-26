# dual_camera_overlap_final.py
# Paste this file and run it. Make sure IDS uEye SDK is installed and cameras are not locked by ThorCam/Camera Manager.

from cam_functions import *

# Try to get screen width via tkinter, otherwise fallback to fixed width
try:
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    root.destroy()
except Exception:
    screen_width = 1200  # fallback if tkinter missing

def main():
    # Quick check number of cameras
    n = ueye.INT()
    require_ok(ueye.is_GetNumberOfCameras(n), "is_GetNumberOfCameras failed")
    if n.value < 2:
        raise RuntimeError(f"Need at least 2 cameras; found {n.value}. Close ThorCam/Camera Manager if they are running.")

    # IMPORTANT: close ThorCam/Camera Manager before running this script
    # Open camera 0 and 1 (0-based indices)
    cam0 = init_camera_by_index(0)
    cam1 = init_camera_by_index(1)
    cams = [cam0, cam1]

    # Setup plot
    overlap_history = deque(maxlen=200)
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 200)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Overlap Integral")

    try:
        while True:
            f0 = grab_frame(cam0)
            f1 = grab_frame(cam1)

            # crop to common size (safe)
            H = min(f0.shape[0], f1.shape[0])
            W = min(f0.shape[1], f1.shape[1])
            f0c = f0[:H, :W]
            f1c = f1[:H, :W]

            ov = overlap_integral(f0c, f1c)
            overlap_history.append(float(ov))

            # side-by-side pad
            hmax = max(f0.shape[0], f1.shape[0])
            def pad(img, hmax):
                if img.shape[0] == hmax:
                    return img
                pad_rows = hmax - img.shape[0]
                return np.vstack([img, np.zeros((pad_rows, img.shape[1], 3), dtype=img.dtype)])
            combined = np.hstack((pad(f0, hmax), pad(f1, hmax)))

            # Resize to screen width while keeping aspect ratio
            h, w = combined.shape[:2]
            if w == 0:
                continue
            scale = float(screen_width) / float(w)
            new_h = max(1, int(h * scale))
            resized = cv2.resize(combined, (screen_width, new_h), interpolation=cv2.INTER_AREA)

            cv2.imshow("Dual uEye (press 'q' to quit)", resized)

            # update plot
            line.set_xdata(np.arange(len(overlap_history)))
            line.set_ydata(list(overlap_history))
            ax.set_xlim(0, max(200, len(overlap_history)))
            plt.pause(0.001)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cleanup_cameras(cams)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
