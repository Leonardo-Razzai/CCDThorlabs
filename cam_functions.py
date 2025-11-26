import os
# If ueye_api.dll is not on PATH, uncomment and set the correct path before importing pyueye:
# os.environ["PATH"] += r";C:\Program Files\IDS\uEye\Develop\Bin"

from pyueye import ueye
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import time


def require_ok(ret, msg):
    if ret != ueye.IS_SUCCESS:
        raise RuntimeError(f"{msg} (ueye error {ret})")

def init_camera_by_index(idx):
    """
    Initialize uEye camera by index (0-based). Returns handle and buffer info.
    """
    hCam = ueye.HIDS(idx)  # 0-based index for the camera
    ret = ueye.is_InitCamera(hCam, None)
    require_ok(ret, f"is_InitCamera failed for index {idx} (ret={ret})")

    # Set color mode to MONO8 (8-bit grayscale)
    require_ok(ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8), "is_SetColorMode MONO8 failed")

    # Get AOI (image size)
    rectAOI = ueye.IS_RECT()
    require_ok(ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI)), "is_AOI get failed")
    width = int(rectAOI.s32Width)
    height = int(rectAOI.s32Height)

    # Allocate image memory (8 bits per pixel)
    mem_ptr = ueye.c_mem_p()
    mem_id  = ueye.int()
    require_ok(ueye.is_AllocImageMem(hCam, width, height, 8, mem_ptr, mem_id), "is_AllocImageMem failed")
    require_ok(ueye.is_SetImageMem(hCam, mem_ptr, mem_id), "is_SetImageMem failed")

    # Optional: inquire pitch (stride). If it fails, we'll use width as fallback.
    pitch = None
    try:
        pitch_i = ueye.INT()
        ueye.is_GetImageMemPitch(hCam, pitch_i)
        pitch = int(pitch_i.value)
        # sanity
        if pitch <= 0:
            pitch = width
    except Exception:
        pitch = width

    # Start live capture (non-blocking)
    require_ok(ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT), "is_CaptureVideo failed")

    return {
        "hCam": hCam,
        "mem_ptr": mem_ptr,
        "mem_id": mem_id,
        "width": width,
        "height": height,
        "pitch": pitch
    }

def _get_data_compat(mem_ptr, width, height, bits, pitch):
    """
    pyueye get_data has slightly different signatures across builds.
    Try named param first, then positional fallback.
    Returns a bytes-like object or buffer.
    """
    try:
        return ueye.get_data(mem_ptr, width, height, bits, pitch=pitch, copy=False)
    except TypeError:
        return ueye.get_data(mem_ptr, width, height, bits, pitch, copy=False)

def grab_frame(cam):
    """
    Read current image for camera dict returned by init_camera_by_index.
    Returns a BGR image (uint8).
    """
    mem_ptr = cam["mem_ptr"]
    w = int(cam["width"])
    h = int(cam["height"])
    pitch = int(cam["pitch"])
    bits = 8
    arr = _get_data_compat(mem_ptr, w, h, bits, pitch)
    # arr may be bytes or buffer; make into numpy
    frame = np.frombuffer(arr, dtype=np.uint8)
    # reshape takes pitch as stride
    try:
        frame = frame.reshape((h, pitch))[:, :w]
    except Exception:
        # fallback: reshape to (h, w) if contiguous
        frame = frame.reshape((h, w))
    # Convert to BGR for display consistency
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

def overlap_integral(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    s1 = g1.sum() + 1e-12
    s2 = g2.sum() + 1e-12
    g1 /= s1
    g2 /= s2
    return (np.sum(np.sqrt(g1 * g2)))**2

def cleanup_cameras(cams):
    for cam in cams:
        h = cam["hCam"]
        mem_ptr = cam["mem_ptr"]
        mem_id = cam["mem_id"]
        try:
            ueye.is_StopLiveVideo(h, ueye.IS_FORCE_VIDEO_STOP)
        except Exception:
            pass
        try:
            ueye.is_FreeImageMem(h, mem_ptr, mem_id)
        except Exception:
            pass
        try:
            ueye.is_ExitCamera(h)
        except Exception:
            pass