import sys
import os
import time
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

# --- CONFIGURATION ---
FORCE_SIMULATION = False # Set to False if you have the real camera connected

# --- CAMERA SETUP ---
try:
    from pyueye import ueye
    UEYE_AVAILABLE = True
except ImportError:
    UEYE_AVAILABLE = False

def init_camera_or_sim(idx=0):
    if FORCE_SIMULATION:
        return init_sim_mode()

    if UEYE_AVAILABLE:
        try:
            hCam = ueye.HIDS(idx)
            if ueye.is_InitCamera(hCam, None) == ueye.IS_SUCCESS:
                ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8)
                rect = ueye.IS_RECT()
                ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rect, ueye.sizeof(rect))
                width, height = int(rect.s32Width), int(rect.s32Height)
                mem_ptr = ueye.c_mem_p()
                mem_id  = ueye.int()
                ueye.is_AllocImageMem(hCam, width, height, 8, mem_ptr, mem_id)
                ueye.is_SetImageMem(hCam, mem_ptr, mem_id)
                ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
                print(f"SUCCESS: Connected to uEye Camera {idx}")
                return {"type": "real", "hCam": hCam, "mem_ptr": mem_ptr, 
                        "mem_id": mem_id, "width": width, "height": height}
        except Exception as e:
            print(f"Camera init failed: {e}")

    return init_sim_mode()

def init_sim_mode():
    print("DEBUG: Simulating Zone Plate...")
    w, h = 640, 480
    x = np.linspace(-10, 10, w)
    y = np.linspace(-10 * (h/w), 10 * (h/w), h)
    X, Y = np.meshgrid(x, y)
    return {"type": "sim", "width": w, "height": h, "grid": X**2 + Y**2, "t_start": time.time()}

def grab_frame(cam):
    if cam["type"] == "real":
        try:
            arr = ueye.get_data(cam["mem_ptr"], cam["width"], cam["height"], 8, cam["width"], copy=False)
            return np.frombuffer(arr, dtype=np.uint8).reshape((cam["height"], cam["width"]))
        except:
            return np.zeros((cam["height"], cam["width"]), dtype=np.uint8)
    else:
        # Simulation: Zone Plate + Specific Periodic Noise
        t = time.time() - cam["t_start"]
        # Base image
        img = 128 + 80 * np.cos(cam["grid"]*0.5) 
        # Add a specific high-frequency interference (The target for your notch filter)
        rows, cols = cam["height"], cam["width"]
        y_idx, x_idx = np.indices((rows, cols))
        interference = 40 * np.sin(2 * np.pi * (x_idx * 0.1 + y_idx * 0.1) + t*5)
        
        return np.clip(img + interference, 0, 255).astype(np.uint8)

def cleanup(cam):
    if cam.get("type") == "real" and UEYE_AVAILABLE:
        ueye.is_ExitCamera(cam["hCam"])

# --- UPDATED FILTER LOGIC: NOTCH FILTER ---

def apply_fft_notch(image, r, kx, ky):
    """
    Removes components at (kx, ky) with radius r.
    Everything else is passed through.
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # 1. Compute FFT
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # 2. Create Mask (Initialized to ONES = Pass All)
    mask = np.ones((rows, cols), np.uint8)
    
    # Calculate coordinates from sliders (100 = center)
    # Map 0..200 -> -1.0..+1.0
    dx = int(((kx - 100) / 100.0) * ccol)
    dy = int(((ky - 100) / 100.0) * crow)
    
    center_1 = (ccol + dx, crow + dy)
    center_2 = (ccol - dx, crow - dy)
    
    # 3. Draw Black Circles (0) to block frequencies
    # If radius is small, ensure at least 1px
    if r < 1: r = 1
    cv2.circle(mask, center_1, r, 0, -1) # -1 = filled circle
    cv2.circle(mask, center_2, r, 0, -1) # Symmetry is required for real images!

    # 4. Apply Mask
    fshift = dft_shift * mask
    
    # 5. Inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 6. Visualization
    # Show the full spectrum, but draw a RED CIRCLE where we are blocking
    mag = 20 * np.log(cv2.magnitude(dft_shift.real, dft_shift.imag) + 1)
    spec_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_vis = cv2.cvtColor(spec_vis, cv2.COLOR_GRAY2BGR)
    
    # Draw empty red circles to show user what they are targeting
    cv2.circle(spec_vis, center_1, r, (0, 0, 255), 2)
    cv2.circle(spec_vis, center_2, r, (0, 0, 255), 2)
    
    return img_back, spec_vis

# --- UI HELPERS ---

def draw_plot(data, h=200, w=400):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if len(data) < 2: return img
    norm_data = h - (data * (h / 255.0)).astype(int)
    norm_data = np.clip(norm_data, 0, h-1)
    x_scale = w / len(data)
    pts = []
    for i, y in enumerate(norm_data):
        pts.append((int(i * x_scale), y))
    cv2.polylines(img, [np.array(pts)], False, (0, 255, 0), 1)
    cv2.line(img, (0, h//2), (w, h//2), (50,50,50), 1) 
    return img

def get_line_profile(img, p1, p2):
    num = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
    if num == 0: return np.array([])
    x, y = np.linspace(p1[0], p2[0], num), np.linspace(p1[1], p2[1], num)
    h, w = img.shape
    x = np.clip(x, 0, w-1).astype(int)
    y = np.clip(y, 0, h-1).astype(int)
    return img[y, x]

class AppState:
    def __init__(self):
        self.p1, self.p2 = None, None
        self.dragging = False
        self.cmap_idx = 0
        self.cmaps = [("Gray", None), ("Jet", cv2.COLORMAP_JET), ("Hot", cv2.COLORMAP_HOT)]

# --- MAIN ---

def main():
    cam = init_camera_or_sim()
    state = AppState()
    win = "Camera Lab (Notch Filter)"
    cv2.namedWindow(win)

    def mouse(event, x, y, f, p):
        if x > cam['width']: return 
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging = True; state.p1 = (x, y); state.p2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.p2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False; state.p2 = (x, y)
            
    cv2.setMouseCallback(win, mouse)
    def nothing(x): pass
    
    # Slider Setup
    cv2.createTrackbar("Mode", win, 0, 2, nothing) # 0=Raw, 1=Gauss, 2=Notch
    cv2.createTrackbar("Param1", win, 10, 100, nothing) # Radius/Kernel
    cv2.createTrackbar("K_X", win, 100, 200, nothing) # Frequency X
    cv2.createTrackbar("K_Y", win, 100, 200, nothing) # Frequency Y

    try:
        while True:
            frame = grab_frame(cam)
            mode = cv2.getTrackbarPos("Mode", win)
            p1 = cv2.getTrackbarPos("Param1", win)
            if p1 < 1: p1 = 1
            
            proc = frame.copy()
            spec_vis = None
            
            if mode == 1: # Gaussian
                k = p1 if p1 % 2 == 1 else p1 + 1
                proc = cv2.GaussianBlur(proc, (k, k), 0)
                
            elif mode == 2: # FFT Notch
                kx = cv2.getTrackbarPos("K_X", win)
                ky = cv2.getTrackbarPos("K_Y", win)
                proc, spec_vis = apply_fft_notch(proc, p1, kx, ky)

            # Colorize
            disp = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
            if state.cmaps[state.cmap_idx][1] is not None:
                disp = cv2.applyColorMap(proc, state.cmaps[state.cmap_idx][1])

            # Overlays
            prof_data = np.array([])
            if state.p1 and state.p2:
                cv2.line(disp, state.p1, state.p2, (0, 0, 255), 2)
                prof_data = get_line_profile(proc, state.p1, state.p2)

            # Spectrum Overlay (Top Right)
            if spec_vis is not None:
                h_s, w_s = 150, 150
                spec_vis = cv2.resize(spec_vis, (w_s, h_s))
                # Add "Notch" text label to spectrum
                cv2.putText(spec_vis, "Remove", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                disp[0:h_s, 0:w_s] = spec_vis
                cv2.rectangle(disp, (0,0), (w_s, h_s), (0,255,0), 1)

            # Layout
            plot_panel = draw_plot(prof_data, h=disp.shape[0], w=400)
            final_ui = np.hstack([disp, plot_panel])
            cv2.imshow(win, final_ui)

            # Inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): state.cmap_idx = (state.cmap_idx + 1) % len(state.cmaps)
            elif key == ord('s'):
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                fp = filedialog.asksaveasfilename(defaultextension=".png",
                    filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
                root.destroy()
                if fp:
                    ext = os.path.splitext(fp)[1] or ".png"
                    ok, buf = cv2.imencode(ext, final_ui)
                    if ok: buf.tofile(fp)
                    print(f"Saved to {fp}")
                cv2.waitKey(1)

    except KeyboardInterrupt: pass
    finally:
        cleanup(cam)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()