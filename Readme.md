# Camera Lab: Real-Time FFT & Profile Analysis

A high-performance Python application for real-time image analysis using IDS uEye cameras. This tool features a robust "Pure OpenCV" GUI to prevent freezing, a Simulation Mode for offline testing, and advanced Frequency Domain (FFT) filtering to remove periodic noise.

## 1\. Features

  * **Dual Hardware Support:**
      * **Real Mode:** Connects to IDS uEye cameras (USB/GigE) via `pyueye`.
      * **Simulation Mode:** Generates a dynamic "Zone Plate" with artificial periodic interference to test filters without hardware.
  * **Real-Time Filtering:**
      * **Gaussian Blur:** Spatial filtering for random noise.
      * **FFT Notch Filter:** Frequency domain filtering to surgically remove periodic noise (Moiré patterns, grid interference) by targeting specific frequencies.
  * **Analysis Tools:**
      * **Live Line Profiling:** Draw a line on the video to see a real-time intensity graph (plotted natively, no Matplotlib lag).
      * **Spectrum Visualization:** Picture-in-Picture display of the Fourier Spectrum to guide filter tuning.
  * **Utilities:**
      * **Robust Saving:** Unicode-safe image saving (works with special characters and Windows paths).
      * **False Color:** Cycle through heatmaps (Jet, Hot, Viridis) to visualize intensity variations.

-----

## 2\. Installation

### Requirements

Ensure you have Python 3.8+ installed.

```bash
pip install numpy opencv-python
```

### Hardware Drivers (Optional)

If you are using a physical **IDS uEye Camera**:

1.  Install the **IDS Software Suite** (Drivers) from the IDS website.
2.  Install the Python wrapper:
    ```bash
    pip install pyueye
    ```

*Note: If drivers are missing, the software automatically launches in Simulation Mode.*

-----

## 3\. Configuration

Open the script (`camera_lab.py`) and look at the top configuration block:

```python
# Set to False to connect to a real camera.
# Set to True to force the simulator even if a camera is connected.
FORCE_SIMULATION = True 
```

-----

## 4\. How to Use

Run the script:

```bash
python camera_lab.py
```

### A. The Control Panel (Sliders)

| Slider | Description |
| :--- | :--- |
| **Mode** | **0:** Raw Image<br>**1:** Gaussian Blur<br>**2:** FFT Notch Filter |
| **Param1** | **Gaussian:** Kernel Size (Blur strength)<br>**FFT:** Radius of the removal circle (Notch size) |
| **K\_X** | **FFT Only:** Horizontal position of the frequency blocker (100 = Center/DC) |
| **K\_Y** | **FFT Only:** Vertical position of the frequency blocker (100 = Center/DC) |

### B. Using the FFT Notch Filter (Mode 2)

This mode is designed to remove **periodic noise** (stripes/grids).

1.  Set **Mode** to `2`.
2.  Look at the **Spectrum Overlay** (top-left corner).
      * The center bright dot is the image signal.
      * **Bright stars/dots away from the center** are the noise components.
3.  Adjust **K\_X** and **K\_Y** to move the **Red Circle** until it covers the noise dot.
4.  Adjust **Param1** to change the size of the Red Circle (blocking radius).
5.  The noise stripes should disappear from the main image immediately.

### C. Live Line Profiling

1.  **Click and Drag** your mouse across the camera feed.
2.  A red line will appear on the image.
3.  The panel on the right will display the pixel intensity graph along that line in real-time.

### D. Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| **Q** | Quit the application. |
| **S** | Open "Save As" dialog (Video pauses while saving). |
| **C** | Cycle Color Palettes (Grayscale $\to$ Jet $\to$ Hot). |

-----

## 5\. Troubleshooting

**"The window never opens."**

  * If `FORCE_SIMULATION = False`, the script tries to initialize the camera. If the camera is disconnected or the driver is hung, this can take 10-20 seconds to time out. Check your USB connection or set `FORCE_SIMULATION = True` to debug.

**"The image is too dark/bright."**

  * This software uses the camera's auto-exposure on init. Adjust your physical lens aperture. In simulation mode, the brightness is fixed.

**"I saved an image but can't find it."**

  * Check the console output. The script prints exactly where it saved the file. If you cancelled the dialog, it will say "Save cancelled."