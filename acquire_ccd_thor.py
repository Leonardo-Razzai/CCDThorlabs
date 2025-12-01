import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime

# --- CONFIGURATION ---
FORCE_SIMULATION = False  # Set False if you have a real uEye camera connected

# --- CAMERA SETUP ---
try:
    from pyueye import ueye
    UEYE_AVAILABLE = True
except ImportError:
    UEYE_AVAILABLE = False

def init_camera_or_sim(idx=0):
    if FORCE_SIMULATION: return init_sim_mode()
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
                return {"type": "real", "hCam": hCam, "mem_ptr": mem_ptr, 
                        "mem_id": mem_id, "width": width, "height": height}
        except Exception as e:
            print(f"Camera Error: {e}")
    return init_sim_mode()

def init_sim_mode():
    w, h = 640, 480 
    x = np.linspace(-10, 10, w)
    y = np.linspace(-10 * (h/w), 10 * (h/w), h)
    X, Y = np.meshgrid(x, y)
    return {"type": "sim", "width": w, "height": h, "grid": X**2 + Y**2, "t_start": time.time()}

def grab_frame(cam, exposure_factor=1.0):
    img = None
    if cam["type"] == "real":
        try:
            arr = ueye.get_data(cam["mem_ptr"], cam["width"], cam["height"], 8, cam["width"], copy=False)
            img = np.frombuffer(arr, dtype=np.uint8).reshape((cam["height"], cam["width"]))
        except:
            img = np.zeros((cam["height"], cam["width"]), dtype=np.uint8)
    else:
        t = time.time() - cam["t_start"]
        base = 128 + 80 * np.cos(cam["grid"]*0.5) 
        rows, cols = cam["height"], cam["width"]
        y_idx, x_idx = np.indices((rows, cols))
        inf = 30 * np.sin(2 * np.pi * (x_idx * 0.15 + y_idx * 0.05) + t*4)
        img = np.clip(base + inf, 0, 255).astype(np.uint8)

    if exposure_factor != 1.0:
        img = cv2.multiply(img, exposure_factor)
    return img

def cleanup(cam):
    if cam.get("type") == "real" and UEYE_AVAILABLE:
        ueye.is_ExitCamera(cam["hCam"])

# --- PROCESSING ---
def process_image_and_fft(image, notches, cursor_pos=None, cursor_radius=5):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    mag = 20 * np.log(cv2.magnitude(dft_shift.real, dft_shift.imag) + 1)
    spec_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    mask = np.ones((rows, cols), np.uint8)
    
    for (nx, ny, r) in notches:
        cv2.circle(mask, (nx, ny), r, 0, -1)
        sym_x, sym_y = ccol + (ccol - nx), crow + (crow - ny)
        cv2.circle(mask, (sym_x, sym_y), r, 0, -1)
        
        cv2.circle(spec_vis, (nx, ny), r, 0, -1) 
        cv2.circle(spec_vis, (nx, ny), r, 255, 1)
        cv2.circle(spec_vis, (sym_x, sym_y), r, 0, -1)
        cv2.circle(spec_vis, (sym_x, sym_y), r, 255, 1)

    if cursor_pos:
        cx, cy = cursor_pos
        cv2.circle(spec_vis, (cx, cy), cursor_radius, 200, 1)

    fshift = dft_shift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back, spec_vis

def apply_zoom(img, zoom_level):
    if zoom_level <= 1.0: return img
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    nh, nw = int(h / zoom_level), int(w / zoom_level)
    return cv2.resize(img[max(0, cy-nh//2):min(h, cy+nh//2), max(0, cx-nw//2):min(w, cx+nw//2)], (w, h))

def resize_and_pad(img, target_w, target_h):
    h, w = img.shape[:2]
    # This logic ensures the content keeps its aspect ratio
    # even if the window is resized to a strange shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if len(img.shape) == 3: canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    else: canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
    return canvas, scale, off_x, off_y

def screen_to_orig_zoomed(sx, sy, scale_info, orig_w, orig_h, zoom_level):
    scale, off_x, off_y = scale_info
    rel_x, rel_y = sx - off_x, sy - off_y
    if not (0 <= rel_x < (orig_w * scale) and 0 <= rel_y < (orig_h * scale)):
        return None
    view_x = rel_x / scale
    view_y = rel_y / scale
    if zoom_level <= 1.0: return int(view_x), int(view_y)
    cx, cy = orig_w / 2, orig_h / 2
    offset_x = cx - (orig_w / (2 * zoom_level))
    offset_y = cy - (orig_h / (2 * zoom_level))
    final_x = offset_x + (view_x / zoom_level)
    final_y = offset_y + (view_y / zoom_level)
    return int(final_x), int(final_y)

def orig_to_screen_zoomed(cx, cy, scale_info, orig_w, orig_h, zoom_level):
    scale, off_x, off_y = scale_info
    if zoom_level <= 1.0:
        view_x, view_y = cx, cy
    else:
        center_x, center_y = orig_w / 2, orig_h / 2
        offset_x = center_x - (orig_w / (2 * zoom_level))
        offset_y = center_y - (orig_h / (2 * zoom_level))
        view_x = (cx - offset_x) * zoom_level
        view_y = (cy - offset_y) * zoom_level
    sx = int(view_x * scale) + off_x
    sy = int(view_y * scale) + off_y
    return sx, sy

def draw_graph(data, w, h, color=(0, 255, 0), title=""):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    if len(data) == 0: return bg
    norm = h - 1 - (data * ((h-30) / 255.0)).astype(int) - 15
    x_scale = w / len(data)
    pts = []
    for i, y in enumerate(norm):
        pts.append((int(i * x_scale), y))
    cv2.polylines(bg, [np.array(pts)], False, color, 2)
    cv2.rectangle(bg, (0,0), (w, h), (30, 30, 30), 1)
    cv2.putText(bg, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return bg

def generate_help_screen(w, h):
    """ Creates a Full Screen Black Image with Instructions """
    screen = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(screen, "CAMERA LAB - USER GUIDE", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
    cv2.putText(screen, "(Click anywhere to Close)", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)

    col1_x = 50
    col2_x = w // 2 + 20
    start_y = 220
    line_h = 45

    cv2.putText(screen, "--- TOP-LEFT: CAMERA VIEW ---", (col1_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(screen, "Left Click:  Move Red Crosshair (Profile)", (col1_x, start_y + 1*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    cv2.putText(screen, "Mouse Wheel: Zoom In / Out", (col1_x, start_y + 2*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)

    cv2.putText(screen, "--- TOP-RIGHT: FFT SPECTRUM ---", (col1_x, start_y + 4*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(screen, "Left Click:  Block Frequency (Remove Noise)", (col1_x, start_y + 5*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    cv2.putText(screen, "Right Click: Clear ALL filters", (col1_x, start_y + 6*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    cv2.putText(screen, "Mouse Wheel: Zoom Spectrum", (col1_x, start_y + 7*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)

    cv2.putText(screen, "--- GENERAL CONTROLS ---", (col2_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(screen, "Gain:   Adjust Exposure / Brightness", (col2_x, start_y + 1*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    cv2.putText(screen, "Radius: Size of the Frequency Block Filter", (col2_x, start_y + 2*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    cv2.putText(screen, "[f]  TOGGLE FULLSCREEN / WINDOWED", (col2_x, start_y + 3*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(screen, "[s]  INSTANT SAVE (to /Captures folder)", (col2_x, start_y + 4*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(screen, "[c]  Cycle Pseudo Colors", (col2_x, start_y + 5*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(screen, "[g]  Toggle Red Grid Lines", (col2_x, start_y + 6*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(screen, "[h]  Open/Close This Help", (col2_x, start_y + 7*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(screen, "[q]  Quit Application", (col2_x, start_y + 8*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return screen

# --- STATE & MAIN ---

class AppState:
    def __init__(self, cam_w, cam_h):
        self.cursor_cam = (cam_w // 2, cam_h // 2) 
        self.cursor_fft = None 
        self.zoom_cam = 1.0
        self.zoom_fft = 1.0
        self.cmap_idx = 0
        self.cmaps = [("Gray", None), ("Jet", cv2.COLORMAP_JET), ("Hot", cv2.COLORMAP_HOT)]
        self.notches = [] 
        self.notch_radius = 10 
        self.show_lines = True
        self.cam_scale_info = (1.0, 0, 0)
        self.fft_scale_info = (1.0, 0, 0)
        self.show_help = False
        self.last_save_time = 0
        self.last_save_path = ""
        self.is_fullscreen = True # Track Window Mode

def main():
    # 1. SETUP DIRS
    SAVE_DIR = "Captures"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    cam = init_camera_or_sim()
    CAM_W, CAM_H = cam["width"], cam["height"]
    state = AppState(CAM_W, CAM_H)
    
    win_name = "Lab App"
    
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Placeholder dimensions
    QUAD_W, QUAD_H = 960, 540 
    BTN_RECT = (20, 20, 100, 40)

    def mouse_callback(event, x, y, flags, param):
        if state.show_help:
            if event == cv2.EVENT_LBUTTONDOWN:
                state.show_help = False
            return 

        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = BTN_RECT
            if bx <= x <= bx+bw and by <= y <= by+bh:
                state.show_help = True
                return

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 0.1 if flags > 0 else -0.1
            if x >= QUAD_W and y < QUAD_H:
                state.zoom_fft = np.clip(state.zoom_fft + delta, 1.0, 10.0)
            else:
                state.zoom_cam = np.clip(state.zoom_cam + delta, 1.0, 5.0)

        if event == cv2.EVENT_MOUSEMOVE:
            if x >= QUAD_W and y < QUAD_H:
                res = screen_to_orig_zoomed(x - QUAD_W, y, state.fft_scale_info, CAM_W, CAM_H, state.zoom_fft)
                if res: state.cursor_fft = res
            else:
                state.cursor_fft = None
            if flags & cv2.EVENT_FLAG_LBUTTON and x < QUAD_W and y < QUAD_H:
                res = screen_to_orig_zoomed(x, y, state.cam_scale_info, CAM_W, CAM_H, state.zoom_cam)
                if res: state.cursor_cam = res

        if event == cv2.EVENT_LBUTTONDOWN:
            if x < QUAD_W and y < QUAD_H:
                res = screen_to_orig_zoomed(x, y, state.cam_scale_info, CAM_W, CAM_H, state.zoom_cam)
                if res: state.cursor_cam = res
            elif x >= QUAD_W and y < QUAD_H:
                res = screen_to_orig_zoomed(x - QUAD_W, y, state.fft_scale_info, CAM_W, CAM_H, state.zoom_fft)
                if res: state.notches.append((res[0], res[1], state.notch_radius))

        if event == cv2.EVENT_RBUTTONDOWN and x >= QUAD_W and y < QUAD_H:
            state.notches = []

    cv2.setMouseCallback(win_name, mouse_callback)
    
    def on_radius_change(val): state.notch_radius = max(1, val)
    cv2.createTrackbar("Radius", win_name, 10, 50, on_radius_change)
    cv2.createTrackbar("Gain", win_name, 100, 300, lambda x: None)

    try:
        while True:
            # --- DYNAMIC RESIZING LOGIC ---
            try:
                rect = cv2.getWindowImageRect(win_name)
                if rect is not None and rect[2] > 0:
                    win_w, win_h = rect[2], rect[3]
                    QUAD_W = win_w // 2
                    QUAD_H = win_h // 2
            except: pass

            if state.show_help:
                final_ui = generate_help_screen(QUAD_W * 2, QUAD_H * 2)
                cv2.imshow(win_name, final_ui)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('h') or key == ord('q'): state.show_help = False
                continue

            gain = cv2.getTrackbarPos("Gain", win_name) / 100.0
            raw = grab_frame(cam, gain)
            
            processed, fft_vis = process_image_and_fft(raw, state.notches, state.cursor_fft, state.notch_radius)
            view_cam = apply_zoom(processed, state.zoom_cam)
            view_fft = apply_zoom(fft_vis, state.zoom_fft)

            view_color = cv2.cvtColor(view_cam, cv2.COLOR_GRAY2BGR)
            if state.cmaps[state.cmap_idx][1] is not None:
                view_color = cv2.applyColorMap(view_cam, state.cmaps[state.cmap_idx][1])
            fft_color = cv2.cvtColor(view_fft, cv2.COLOR_GRAY2BGR)

            cx, cy = state.cursor_cam
            cx = np.clip(cx, 0, CAM_W-1); cy = np.clip(cy, 0, CAM_H-1)
            row_data = processed[cy, :]; col_data = processed[:, cx]

            disp_cam, scale_c, off_xc, off_yc = resize_and_pad(view_color, QUAD_W, QUAD_H)
            state.cam_scale_info = (scale_c, off_xc, off_yc)
            
            if state.show_lines:
                sx, sy = orig_to_screen_zoomed(cx, cy, state.cam_scale_info, CAM_W, CAM_H, state.zoom_cam)
                cv2.line(disp_cam, (0, sy), (QUAD_W, sy), (0, 0, 255), 1)
                cv2.line(disp_cam, (sx, 0), (sx, QUAD_H), (0, 0, 255), 1)
                cv2.circle(disp_cam, (sx, sy), 5, (0, 0, 255), -1)

            disp_fft, scale_f, off_xf, off_yf = resize_and_pad(fft_color, QUAD_W, QUAD_H)
            state.fft_scale_info = (scale_f, off_xf, off_yf)
            z_txt = f"{state.zoom_fft:.1f}x" if state.zoom_fft > 1.0 else "1.0x"
            cv2.putText(disp_fft, f"FFT (Zoom: {z_txt})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            disp_plot_x = draw_graph(row_data, QUAD_W, QUAD_H, (0, 255, 255), f"H-Profile (y={cy})")
            disp_plot_y = draw_graph(col_data, QUAD_W, QUAD_H, (255, 0, 255), f"V-Profile (x={cx})")

            final_ui = np.vstack([np.hstack([disp_cam, disp_fft]), np.hstack([disp_plot_x, disp_plot_y])])

            bx, by, bw, bh = BTN_RECT
            cv2.rectangle(final_ui, (bx, by), (bx+bw, by+bh), (80, 80, 80), -1)
            cv2.rectangle(final_ui, (bx, by), (bx+bw, by+bh), (200, 200, 200), 1)
            cv2.putText(final_ui, "HELP", (bx+20, by+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if time.time() - state.last_save_time < 3.0:
                cv2.putText(final_ui, f"SAVED: {state.last_save_path}", (bx + 120, by+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(win_name, final_ui)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): state.cmap_idx = (state.cmap_idx + 1) % len(state.cmaps)
            elif key == ord('g'): state.show_lines = not state.show_lines
            elif key == ord('h'): state.show_help = not state.show_help
            elif key == ord('s'):
                # INSTANT SAVE
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"img_{ts}.png"
                filepath = os.path.join(SAVE_DIR, filename)
                
                success = cv2.imwrite(filepath, view_color)
                if success:
                    state.last_save_time = time.time()
                    state.last_save_path = filename
                    print(f"Saved: {filepath}")
                else:
                    print("Error: Could not save image.")
            
            # --- FULLSCREEN TOGGLE ---
            elif key == ord('f'):
                state.is_fullscreen = not state.is_fullscreen
                if state.is_fullscreen:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, 1280, 720) # Set a Default Window size

    except KeyboardInterrupt: pass
    finally:
        cleanup(cam)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()