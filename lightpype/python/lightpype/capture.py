import sys
import cv2
import numpy as np
import pyqtgraph as pg
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QCheckBox, QSlider, QLabel, QGroupBox, QDoubleSpinBox, QSpinBox, QFileDialog,
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QProgressBar, QLineEdit, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.linear_model import RANSACRegressor

# Known emission lines for common light sources (nm)
REFERENCE_LINES = {
    "Fluorescent": [404.7, 435.8, 546.1, 577.0, 579.0],
    "Mercury": [365.0, 404.7, 435.8, 546.1, 577.0, 579.0],
    "LED": [450, 520, 630],  # Typical RGB LED
    "Sodium": [589.0, 589.6],
    "Neon": [540.1, 585.2, 588.2, 603.0, 616.4, 621.7, 626.6, 633.4, 638.3, 640.2],
    "Custom": []
}


class HDRCaptureThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    failed = pyqtSignal(str)

    def __init__(self, spectrometer, exposures):
        super().__init__()
        self.spectrometer = spectrometer
        self.exposures = exposures

    def run(self):
        try:
            profiles = []
            for i, exp in enumerate(self.exposures):
                self.spectrometer.set_exposure(exp)
                self.progress.emit(int((i + 1) * 100 / len(self.exposures)))

                # Capture frame with retries
                frame = None
                for _ in range(3):
                    frame = self.spectrometer.capture_frame()
                    if frame is not None:
                        break

                if frame is None:
                    self.failed.emit("Failed to capture frame")
                    return

                profile = self.spectrometer.extract_profile(frame)
                profiles.append(profile)

            # Combine profiles avoiding clipped regions
            combined = np.zeros_like(profiles[0])
            valid_counts = np.zeros_like(profiles[0])

            for profile in profiles:
                # Identify non-clipped regions
                non_clipped = (profile < self.spectrometer.clipping_limit) & (profile > 10)
                combined[non_clipped] += profile[non_clipped]
                valid_counts[non_clipped] += 1

            # Avoid division by zero
            valid_counts[valid_counts == 0] = 1
            combined = combined / valid_counts

            # For clipped regions, take the highest exposure value
            for profile in reversed(profiles):
                clipped = (profile >= self.spectrometer.clipping_limit) | (profile <= 10)
                combined[clipped] = profile[clipped]

            self.finished.emit(combined)
        except Exception as e:
            self.failed.emit(f"HDR Capture Error: {str(e)}")


class CalibrationDialog(QDialog):
    def __init__(self, wavelengths, intensities, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.setGeometry(200, 200, 1000, 700)
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d30;
                color: #ffffff;
            }
            QComboBox, QLabel, QPushButton {
                background-color: #3e3e42;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3e3e42;
                color: #ffffff;
            }
        """)

        layout = QVBoxLayout()

        # Plot for spectrum and detected peaks
        self.figure = plt.figure(facecolor='#2d2d30')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        layout.addWidget(self.canvas)

        # Form for selecting reference lines
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFormAlignment(Qt.AlignHCenter | Qt.AlignTop)
        form_layout.setVerticalSpacing(10)

        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(200)
        self.source_combo.addItems(list(REFERENCE_LINES.keys()))
        form_layout.addRow(QLabel("Reference Source:"), self.source_combo)

        # Peak selection with custom wavelengths
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel("Detected Peak"), 0, 0)
        grid_layout.addWidget(QLabel("Pixel Position"), 0, 1)
        grid_layout.addWidget(QLabel("Wavelength (nm)"), 0, 2)

        self.peak_combos = []
        self.pixel_labels = []
        self.wavelength_inputs = []

        for i in range(5):  # Support up to 5 peaks
            peak_combo = QComboBox()
            pixel_label = QLabel("")
            wavelength_input = QLineEdit()
            wavelength_input.setPlaceholderText("Enter wavelength")

            grid_layout.addWidget(peak_combo, i + 1, 0)
            grid_layout.addWidget(pixel_label, i + 1, 1)
            grid_layout.addWidget(wavelength_input, i + 1, 2)

            self.peak_combos.append(peak_combo)
            self.pixel_labels.append(pixel_label)
            self.wavelength_inputs.append(wavelength_input)

        form_layout.addRow(QLabel("Peak Mapping:"), grid_layout)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QPushButton {
                min-width: 80px;
                padding: 5px;
            }
        """)

        layout.addLayout(form_layout)
        layout.addWidget(button_box)
        self.setLayout(layout)

        # Store data
        self.wavelengths = wavelengths
        self.intensities = intensities
        self.peaks = []

        # Find peaks
        self.find_peaks()
        self.update_peak_options()
        self.update_plot()

    def find_peaks(self):
        """Find significant peaks in the spectrum"""
        # Smooth data to reduce noise
        smoothed = savgol_filter(self.intensities, 21, 3)

        # Find peaks with minimum prominence
        min_prominence = np.max(smoothed) * 0.1
        self.peaks, properties = find_peaks(
            smoothed,
            height=min_prominence,
            prominence=min_prominence,
            distance=10,
            width=5
        )

        # Sort peaks by prominence
        if len(self.peaks) > 0:
            prominences = properties['prominences']
            sorted_indices = np.argsort(prominences)[::-1]
            self.peaks = self.peaks[sorted_indices]
            self.peak_properties = {
                'prominences': prominences[sorted_indices],
                'widths': properties['widths'][sorted_indices],
                'heights': properties['peak_heights'][sorted_indices]
            }

    def update_peak_options(self):
        """Populate peak selection dropdowns"""
        for i, combo in enumerate(self.peak_combos):
            combo.clear()
            if i < len(self.peaks):
                peak_idx = self.peaks[i]
                combo.addItem(f"Peak {i + 1} (px: {peak_idx})")
                self.pixel_labels[i].setText(str(peak_idx))
                self.wavelength_inputs[i].clear()
            else:
                combo.setEnabled(False)
                self.pixel_labels[i].setText("")
                self.wavelength_inputs[i].setEnabled(False)

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor='#1e1e1e')

        # Set plot colors for dark theme
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

        # Plot spectrum
        ax.plot(self.wavelengths, self.intensities, 'c-', label='Spectrum')

        # Plot peaks
        if len(self.peaks) > 0:
            ax.plot(self.wavelengths[self.peaks], self.intensities[self.peaks],
                    'ro', markersize=8, label='Detected Peaks')

            # Annotate peaks with their indices
            for i, peak in enumerate(self.peaks):
                ax.annotate(f"{i + 1}",
                            (self.wavelengths[peak], self.intensities[peak]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            color='white',
                            fontsize=10)

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title('Detected Peaks for Calibration')
        ax.legend(facecolor='#3e3e42', edgecolor='none', labelcolor='white')
        ax.grid(True, color='#555555', linestyle='--')
        self.canvas.draw()

    def get_calibration_data(self):
        """Return calibration pairs (wavelength, pixel)"""
        calibration_pairs = []
        for i in range(min(5, len(self.peaks))):
            if self.wavelength_inputs[i].text():
                try:
                    wavelength = float(self.wavelength_inputs[i].text())
                    pixel = self.peaks[i]
                    calibration_pairs.append((wavelength, pixel))
                except ValueError:
                    continue
        return calibration_pairs


class Spectrometer:
    def __init__(self, camera_index=0, config_file="spectrometer_config.json"):
        self.config_file = config_file
        self.load_config()

        # Initialize camera
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        # Apply config settings
        self.apply_camera_settings()

        # Data storage
        self.data_array = None
        self.data_array_peak = None
        self.reference_spectrum = None
        self.dark_frame = None
        self.hdr_profile = None
        self.calibration_coeffs = None
        self.calibration_r2 = 0.0

        # **Add missing flag**
        self.geo_calibrate = False          # Auto geometry calibration disabled by default

        self.running = True
        self.frame = None

    def load_config(self):
        """Load configuration from JSON file"""
        self.config = {
            "camera_index": 0,
            "exposure": -1,
            "gain": 0,
            "brightness": 50,
            "contrast": 50,
            "saturation": 50,
            "sample_height": 360,
            "window_height": 10,
            "smoothing": 10,
            "clipping_limit": 240,
            "calibration": {
                "pixels": [],
                "wavelengths": []
            },
            "hdr_exposures": [10, 30, 60, 100],
            "dark_frame_path": "",
            "reference_path": ""
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with default config
                    self.config.update(loaded_config)

                    # Load dark frame if path exists
                    if self.config["dark_frame_path"] and os.path.exists(self.config["dark_frame_path"]):
                        self.dark_frame = np.load(self.config["dark_frame_path"])

                    # Load reference spectrum if path exists
                    if self.config["reference_path"] and os.path.exists(self.config["reference_path"]):
                        self.reference_spectrum = np.load(self.config["reference_path"])
            except Exception as e:
                print(f"Error loading config: {str(e)}")
                # Reset to default on error
                self.config = self.config.copy()

    def save_config(self):
        """Save configuration to JSON file"""
        # Update config with current settings
        self.config["exposure"] = self.exposure
        self.config["gain"] = self.gain
        self.config["brightness"] = self.brightness
        self.config["contrast"] = self.contrast
        self.config["saturation"] = self.saturation
        self.config["sample_height"] = self.sample_height
        self.config["window_height"] = self.window_height
        self.config["smoothing"] = self.smoothing
        self.config["clipping_limit"] = self.clipping_limit

        # Save dark frame
        if self.dark_frame is not None:
            dark_path = os.path.join(os.path.dirname(self.config_file), "dark_frame.npy")
            np.save(dark_path, self.dark_frame)
            self.config["dark_frame_path"] = dark_path

        # Save reference spectrum
        if self.reference_spectrum is not None:
            ref_path = os.path.join(os.path.dirname(self.config_file), "reference_spectrum.npy")
            np.save(ref_path, self.reference_spectrum)
            self.config["reference_path"] = ref_path

        # Save calibration
        if self.calibration_coeffs is not None:
            self.config["calibration"] = {
                "coeffs": self.calibration_coeffs.tolist(),
                "r2": self.calibration_r2
            }

        # Write to file
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            return False

    def apply_camera_settings(self):
        """Apply loaded settings to camera"""
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.set_exposure(self.config["exposure"])
        self.set_gain(self.config["gain"])
        self.set_brightness(self.config["brightness"])
        self.set_contrast(self.config["contrast"])
        self.set_saturation(self.config["saturation"])

        # Apply other settings
        self.sample_height = self.config["sample_height"]
        self.window_height = self.config["window_height"]
        self.ifactor = self.config["smoothing"] / 100.0
        self.clipping_limit = self.config["clipping_limit"]

        # Apply calibration if exists
        if "calibration" in self.config and "coeffs" in self.config["calibration"]:
            self.calibration_coeffs = np.array(self.config["calibration"]["coeffs"])
            self.calibration_r2 = self.config["calibration"].get("r2", 0.0)

    def capture_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.frame = frame
            return frame
        return None

    def set_exposure(self, exposure):
        self.exposure = exposure
        if exposure > 0:
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.manual_exposure = True
        else:
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.manual_exposure = False

    def set_gain(self, gain):
        self.gain = gain
        self.camera.set(cv2.CAP_PROP_GAIN, gain)

    def set_brightness(self, value):
        self.brightness = value
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, value)

    def set_contrast(self, value):
        self.contrast = value
        self.camera.set(cv2.CAP_PROP_CONTRAST, value)

    def set_saturation(self, value):
        self.saturation = value
        self.camera.set(cv2.CAP_PROP_SATURATION, value)

    def extract_profile(self, frame):
        if frame is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate vertical slice bounds
        start_line = max(0, self.sample_height - self.window_height // 2)
        end_line = min(gray.shape[0], self.sample_height + self.window_height // 2 + 1)

        # Average the vertical window
        profile = np.mean(gray[start_line:end_line, :], axis=0)

        # Apply temporal filtering
        if self.data_array is None:
            self.data_array = profile.astype(np.float32)
            self.data_array_peak = np.copy(self.data_array)
        else:
            self.data_array = self.ifactor * self.data_array + (1 - self.ifactor) * profile
            self.data_array_peak = 0.9 * self.data_array_peak + 0.1 * profile

        return self.data_array

    def pix_to_wave(self, pixel):
        """Convert pixel position to wavelength using calibration"""
        if self.calibration_coeffs is None:
            # Default linear calibration if no calibration exists
            return 400 + (pixel / 1280) * 300  # 400-700nm range

        # Polynomial calibration: wave = c0 + c1*x + c2*x² + ...
        return np.polyval(self.calibration_coeffs, pixel)

    def wave_to_pix(self, wavelength):
        """Convert wavelength to pixel position"""
        if self.calibration_coeffs is None:
            # Inverse of default calibration
            return (wavelength - 400) * 1280 / 300

        # Find pixel where wavelength is closest
        pixels = np.arange(1280)
        wavelengths = self.pix_to_wave(pixels)
        return np.argmin(np.abs(wavelengths - wavelength))

    def find_peaks(self, data, height=10, distance=10):
        """Find peaks in spectral data"""
        peaks, _ = find_peaks(data, height=height, distance=distance)
        return peaks

    def capture_dark_frame(self):
        """Capture background noise reference"""
        frame = self.capture_frame()
        if frame is not None:
            self.dark_frame = self.extract_profile(frame)
            return True
        return False

    def capture_reference(self):
        """Capture reference spectrum"""
        frame = self.capture_frame()
        if frame is not None:
            self.reference_spectrum = self.extract_profile(frame)
            return True
        return False

    def calculate_reflectance(self):
        """Calculate reflectance from reference"""
        if self.reference_spectrum is None or self.data_array is None:
            return None

        # Subtract dark frame if available
        sample = self.data_array
        if self.dark_frame is not None:
            sample = sample - self.dark_frame
            reference = self.reference_spectrum - self.dark_frame
        else:
            reference = self.reference_spectrum

        # Avoid division by zero
        reference = np.clip(reference, 1, None)
        reflectance = np.clip(sample / reference, 0, 1)

        return reflectance

    def calibrate_geometry(self, intensities):
        """Automatically detect rainbow pattern and set calibration points"""
        # Smooth data to reduce noise
        smoothed = savgol_filter(intensities, 21, 3)

        # Find regions with significant signal
        threshold = np.max(smoothed) * 0.1
        signal_regions = np.where(smoothed > threshold)[0]

        if len(signal_regions) == 0:
            return False

        # Find start and end of rainbow
        start_pixel = signal_regions[0]
        end_pixel = signal_regions[-1]

        # Set ROI to rainbow region
        self.minX = max(0, start_pixel - 10)
        self.maxX = min(len(intensities) - 1, end_pixel + 10)

        return True

    def calibrate_wavelength(self, pixels, wavelengths):
        """Calibrate wavelength using reference points"""
        if len(pixels) < 2:
            return False

        # Fit polynomial (cubic by default)
        degree = min(3, len(pixels) - 1)
        coeffs, residuals, _, _, _ = np.polyfit(pixels, wavelengths, degree, full=True)

        # Calculate R²
        p = np.poly1d(coeffs)
        yhat = p(pixels)
        ybar = np.mean(wavelengths)
        ssreg = np.sum((yhat - ybar) ** 2)
        sstot = np.sum((wavelengths - ybar) ** 2)
        r2 = ssreg / sstot if sstot != 0 else 1.0

        # Store calibration
        self.calibration_coeffs = coeffs
        self.calibration_r2 = r2

        return True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrometer")
        self.setGeometry(100, 100, 1400, 900)

        # Load style
        self.load_style()

        # Create spectrometer instance
        self.config_file = "spectrometer_config.json"
        self.spectrometer = None
        self.available_cameras = self.detect_cameras()

        if self.available_cameras:
            self.camera_index = self.available_cameras[0]
            self.spectrometer = Spectrometer(self.camera_index, self.config_file)
        else:
            self.camera_index = -1

        # Create UI
        self.init_ui()

        # Setup update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  # ~30 FPS

    def load_style(self):
        """Set dark theme with light text"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2d2d30;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 5px;
                background-color: transparent;
                color: #a0a0a0;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #3e3e42;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4c4c50;
            }
            QPushButton:pressed {
                background-color: #2a2a2c;
            }
            QPushButton:disabled {
                background-color: #2d2d30;
                color: #777777;
            }
            QComboBox {
                background-color: #3e3e42;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView {
                background-color: #3e3e42;
                color: #ffffff;
                selection-background-color: #0078d7;
            }
            QSlider::groove:horizontal {
                background: #3e3e42;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #a0a0a0;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::add-page:horizontal {
                background: #555555;
            }
            QSlider::sub-page:horizontal {
                background: #0078d7;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                background: #3e3e42;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
                width: 10px;
            }
            QLineEdit {
                background-color: #3e3e42;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 4px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3e3e42;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 3px;
            }
        """)

    def detect_cameras(self):
        """Detect available cameras"""
        cameras = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras

    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left panel - controls and camera view
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        # Camera selection
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))

        self.camera_combo = QComboBox()
        for i, cam_idx in enumerate(self.available_cameras):
            self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
        self.camera_combo.setCurrentIndex(0)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_cameras)

        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(refresh_btn)
        left_panel.addLayout(camera_layout)

        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.camera_label)

        # Camera controls group
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QGridLayout()
        controls_group.setLayout(controls_layout)
        left_panel.addWidget(controls_group)

        # Create control functions with labels
        def create_slider_control(label, min_val, max_val, value, callback, step=None, show_value=True):
            row = controls_layout.rowCount()
            controls_layout.addWidget(QLabel(label), row, 0)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(value)
            slider.valueChanged.connect(callback)
            controls_layout.addWidget(slider, row, 1)

            if show_value:
                value_label = QLabel(str(value))
                slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
                controls_layout.addWidget(value_label, row, 2)

            return slider

        # Exposure controls
        self.exposure_slider = create_slider_control("Exposure:", -13, -1, -7, self.set_exposure)
        self.manual_exposure_cb = QCheckBox("Manual Exposure")
        controls_layout.addWidget(self.manual_exposure_cb, controls_layout.rowCount(), 0, 1, 2)

        # Gain control
        self.gain_slider = create_slider_control("Gain:", 0, 100, 0, self.set_gain)

        # Image controls
        self.brightness_slider = create_slider_control("Brightness:", 0, 100, 50, self.set_brightness)
        self.contrast_slider = create_slider_control("Contrast:", 0, 100, 50, self.set_contrast)
        self.saturation_slider = create_slider_control("Saturation:", 0, 100, 50, self.set_saturation)

        # Sample height control
        self.height_slider = create_slider_control("Sample Height:", 0, 720, 360, self.set_sample_height)

        # Window height control
        self.window_slider = create_slider_control("Window Height:", 1, 100, 10, self.set_window_height)

        # Smoothing factor
        self.smooth_slider = create_slider_control("Smoothing:", 0, 100, 10, self.set_smoothing)

        # Clipping threshold
        self.clipping_slider = create_slider_control("Clipping Threshold:", 0, 255, 240, self.set_clipping_threshold)

        # Action buttons
        buttons_layout = QHBoxLayout()
        self.dark_button = QPushButton("Capture Dark")
        self.ref_button = QPushButton("Capture Ref")
        self.hdr_button = QPushButton("HDR Capture")
        self.calibrate_btn = QPushButton("Calibrate")
        self.save_button = QPushButton("Save Config")
        self.load_button = QPushButton("Load Config")

        buttons_layout.addWidget(self.dark_button)
        buttons_layout.addWidget(self.ref_button)
        buttons_layout.addWidget(self.hdr_button)
        buttons_layout.addWidget(self.calibrate_btn)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.load_button)

        controls_layout.addLayout(buttons_layout, controls_layout.rowCount(), 0, 1, 3)

        # Mode controls
        mode_layout = QHBoxLayout()
        self.absorption_cb = QCheckBox("Absorption Mode")
        self.peak_cb = QCheckBox("Show Peaks")
        self.colorbar_cb = QCheckBox("Show Colorbar")
        self.geocalibrate_cb = QCheckBox("Auto Geometry")

        mode_layout.addWidget(self.absorption_cb)
        mode_layout.addWidget(self.peak_cb)
        mode_layout.addWidget(self.colorbar_cb)
        mode_layout.addWidget(self.geocalibrate_cb)

        controls_layout.addLayout(mode_layout, controls_layout.rowCount(), 0, 1, 3)

        # Right panel - plots
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 2)

        # Spectrum plot
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground('#1e1e1e')
        self.spectrum_plot.setLabel('left', 'Intensity')
        self.spectrum_plot.setLabel('bottom', 'Wavelength (nm)')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_plot.setYRange(0, 255)
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen('y', width=2))
        self.spectrum_peaks = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r'), brush=pg.mkBrush('r'))
        self.spectrum_plot.addItem(self.spectrum_peaks)

        # Add colorbar
        self.colorbar = pg.GradientLegend(size=(100, 15), offset=(10, 10))
        self.colorbar.setGradient(pg.ColorMap([0, 1], [(0, 0, 0), (255, 255, 255)]))
        self.colorbar.setLabels({0: '400nm', 1: '700nm'})
        self.spectrum_plot.addItem(self.colorbar)
        self.colorbar.hide()

        # Reflectance plot
        self.reflectance_plot = pg.PlotWidget()
        self.reflectance_plot.setBackground('#1e1e1e')
        self.reflectance_plot.setLabel('left', 'Reflectance')
        self.reflectance_plot.setLabel('bottom', 'Wavelength (nm)')
        self.reflectance_plot.showGrid(x=True, y=True, alpha=0.3)
        self.reflectance_plot.setYRange(0, 1.2)
        self.reflectance_curve = self.reflectance_plot.plot(pen=pg.mkPen('g', width=2))

        # Calibration info
        calib_label = QLabel("Calibration: Not calibrated")
        calib_label.setStyleSheet("font-weight: bold; color: #ff9900;")
        self.calib_label = calib_label

        right_panel.addWidget(self.spectrum_plot, 3)
        right_panel.addWidget(self.reflectance_plot, 3)
        right_panel.addWidget(calib_label, 1)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        # Connect signals
        self.exposure_slider.valueChanged.connect(self.set_exposure)
        self.gain_slider.valueChanged.connect(self.set_gain)
        self.height_slider.valueChanged.connect(self.set_sample_height)
        self.window_slider.valueChanged.connect(self.set_window_height)
        self.smooth_slider.valueChanged.connect(self.set_smoothing)
        self.clipping_slider.valueChanged.connect(self.set_clipping_threshold)
        self.manual_exposure_cb.stateChanged.connect(self.toggle_manual_exposure)
        self.dark_button.clicked.connect(self.capture_dark)
        self.ref_button.clicked.connect(self.capture_reference)
        self.hdr_button.clicked.connect(self.capture_hdr)
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.save_button.clicked.connect(self.save_config)
        self.load_button.clicked.connect(self.load_config)
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        self.absorption_cb.stateChanged.connect(self.toggle_absorption_mode)
        self.peak_cb.stateChanged.connect(self.toggle_peak_display)
        self.colorbar_cb.stateChanged.connect(self.toggle_colorbar)
        self.geocalibrate_cb.stateChanged.connect(self.toggle_geocalibrate)

        # Initialize plots
        self.x_data = np.arange(1280)
        if self.spectrometer:
            self.wavelengths = self.spectrometer.pix_to_wave(self.x_data)
            self.update_calibration_label()
        else:
            self.wavelengths = self.x_data

    def set_exposure(self, value):
        if self.spectrometer:
            self.spectrometer.set_exposure(value)

    def set_gain(self, value):
        if self.spectrometer:
            self.spectrometer.set_gain(value)

    def set_brightness(self, value):
        if self.spectrometer:
            self.spectrometer.set_brightness(value)

    def set_contrast(self, value):
        if self.spectrometer:
            self.spectrometer.set_contrast(value)

    def set_saturation(self, value):
        if self.spectrometer:
            self.spectrometer.set_saturation(value)

    def set_sample_height(self, value):
        if self.spectrometer:
            self.spectrometer.sample_height = value

    def set_window_height(self, value):
        if self.spectrometer:
            self.spectrometer.window_height = value

    def set_smoothing(self, value):
        if self.spectrometer:
            self.spectrometer.ifactor = value / 100.0

    def set_clipping_threshold(self, value):
        if self.spectrometer:
            self.spectrometer.clipping_limit = value

    def toggle_manual_exposure(self, state):
        if self.spectrometer:
            if state == Qt.Checked:
                self.spectrometer.manual_exposure = True
            else:
                self.spectrometer.manual_exposure = False

    def toggle_absorption_mode(self, state):
        if self.spectrometer:
            self.spectrometer.absorption_mode = (state == Qt.Checked)

    def toggle_peak_display(self, state):
        if self.spectrometer:
            self.spectrometer.show_peaks = (state == Qt.Checked)

    def toggle_colorbar(self, state):
        if self.spectrometer:
            self.spectrometer.show_colorbar = (state == Qt.Checked)
            if state == Qt.Checked:
                self.colorbar.show()
            else:
                self.colorbar.hide()

    def toggle_geocalibrate(self, state):
        if self.spectrometer:
            self.spectrometer.geo_calibrate = (state == Qt.Checked)

    def capture_dark(self):
        if self.spectrometer:
            if self.spectrometer.capture_dark_frame():
                self.status_label.setText("Dark frame captured")
            else:
                self.status_label.setText("Failed to capture dark frame")

    def capture_reference(self):
        if self.spectrometer:
            if self.spectrometer.capture_reference():
                self.status_label.setText("Reference spectrum captured")
            else:
                self.status_label.setText("Failed to capture reference")

    def capture_hdr(self):
        if not self.spectrometer:
            return

        # Create HDR dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("HDR Capture")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout()

        # Exposure settings
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposures (ms):"))
        self.exp_input = QLineEdit(",".join(map(str, self.spectrometer.config["hdr_exposures"])))
        exp_layout.addWidget(self.exp_input)
        layout.addLayout(exp_layout)

        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        layout.addWidget(progress)

        # Status label
        status_label = QLabel("Ready to start capture")
        layout.addWidget(status_label)

        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        dialog.setLayout(layout)

        def start_hdr():
            try:
                exposures = [int(x.strip()) for x in self.exp_input.text().split(",") if x.strip()]
                if not exposures:
                    status_label.setText("No exposures specified")
                    return

                # Start HDR capture thread
                self.hdr_thread = HDRCaptureThread(self.spectrometer, exposures)
                self.hdr_thread.progress.connect(progress.setValue)
                self.hdr_thread.finished.connect(
                    lambda profile: self.hdr_capture_finished(profile, dialog)
                )
                self.hdr_thread.failed.connect(
                    lambda msg: status_label.setText(msg)
                )
                status_label.setText("Capturing...")
                self.hdr_thread.start()
            except Exception as e:
                status_label.setText(f"Error: {str(e)}")

        btn_box.accepted.disconnect()
        btn_box.accepted.connect(start_hdr)

        dialog.exec_()

    def hdr_capture_finished(self, hdr_profile, dialog):
        self.spectrometer.hdr_profile = hdr_profile
        self.spectrometer.data_array = hdr_profile
        dialog.accept()
        self.status_label.setText("HDR capture completed")

    def calibrate(self):
        if not self.spectrometer or self.spectrometer.data_array is None:
            return

        # Capture current spectrum
        wavelengths = self.spectrometer.pix_to_wave(np.arange(len(self.spectrometer.data_array)))
        intensities = self.spectrometer.data_array

        # Show calibration dialog
        dialog = CalibrationDialog(wavelengths, intensities, self)
        if dialog.exec_() == QDialog.Accepted:
            calibration_pairs = dialog.get_calibration_data()
            if calibration_pairs:
                pixels = [pair[1] for pair in calibration_pairs]
                wave_values = [pair[0] for pair in calibration_pairs]

                if self.spectrometer.calibrate_wavelength(pixels, wave_values):
                    self.update_calibration_label()
                    self.status_label.setText(f"Calibration complete (R²={self.spectrometer.calibration_r2:.4f})")
                else:
                    self.status_label.setText("Calibration failed - not enough points")
            else:
                self.status_label.setText("No calibration points selected")

    def change_camera(self, index):
        if index < 0 or index >= len(self.available_cameras):
            return

        new_index = self.camera_combo.itemData(index)
        if new_index == self.camera_index:
            return

        # Recreate spectrometer with new camera
        self.camera_index = new_index
        if self.spectrometer:
            self.spectrometer.camera.release()

        self.spectrometer = Spectrometer(self.camera_index, self.config_file)
        self.status_label.setText(f"Switched to camera {self.camera_index}")

    def refresh_cameras(self):
        self.available_cameras = self.detect_cameras()
        self.camera_combo.clear()
        for i, cam_idx in enumerate(self.available_cameras):
            self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)

        if self.camera_index not in self.available_cameras and self.available_cameras:
            self.camera_index = self.available_cameras[0]
            self.spectrometer = Spectrometer(self.camera_index, self.config_file)
            self.status_label.setText(f"Refreshed cameras, found {len(self.available_cameras)}")
            self.camera_combo.setCurrentIndex(0)
        elif not self.available_cameras:
            self.spectrometer = None
            self.status_label.setText("No cameras detected")

    def save_config(self):
        if self.spectrometer and self.spectrometer.save_config():
            self.status_label.setText(f"Configuration saved to {self.spectrometer.config_file}")
        else:
            self.status_label.setText("Failed to save configuration")

    def load_config(self):
        if self.spectrometer:
            self.spectrometer.load_config()
            self.spectrometer.apply_camera_settings()
            self.update_calibration_label()

            # Update UI controls to match loaded config
            self.exposure_slider.setValue(self.spectrometer.exposure)
            self.gain_slider.setValue(self.spectrometer.gain)
            self.brightness_slider.setValue(self.spectrometer.brightness)
            self.contrast_slider.setValue(self.spectrometer.contrast)
            self.saturation_slider.setValue(self.spectrometer.saturation)
            self.height_slider.setValue(self.spectrometer.sample_height)
            self.window_slider.setValue(self.spectrometer.window_height)
            self.smooth_slider.setValue(int(self.spectrometer.ifactor * 100))
            self.clipping_slider.setValue(self.spectrometer.clipping_limit)

            self.status_label.setText(f"Configuration loaded from {self.spectrometer.config_file}")

    def update_calibration_label(self):
        if self.spectrometer and self.spectrometer.calibration_coeffs is not None:
            degree = len(self.spectrometer.calibration_coeffs) - 1
            text = f"Calibration: {degree}-degree polynomial (R²={self.spectrometer.calibration_r2:.4f})"
            self.calib_label.setText(text)
        else:
            self.calib_label.setText("Calibration: Not calibrated")

    def update(self):
        if not self.spectrometer:
            # Show placeholder if no camera
            pixmap = QPixmap(640, 480)
            pixmap.fill(Qt.black)
            painter = QPainter(pixmap)
            painter.setPen(Qt.white)
            painter.setFont(QFont("Arial", 20))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "No Camera Detected")
            painter.end()
            self.camera_label.setPixmap(pixmap)
            return

        # Capture and process frame
        frame = self.spectrometer.capture_frame()
        if frame is None:
            return

        # Auto geometry calibration
        if getattr(self.spectrometer, 'geo_calibrate', False):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            profile = np.mean(gray, axis=0)
            self.spectrometer.calibrate_geometry(profile)

        # Extract profile
        profile = self.spectrometer.extract_profile(frame)

        # Update camera display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)

        # Draw sample line
        painter = QPainter(q_img)
        painter.setPen(QPen(Qt.red, 2))
        y = getattr(self.spectrometer, 'sample_height', 0)
        painter.drawLine(0, y, width, y)

        # Draw window
        window_height = getattr(self.spectrometer, 'window_height', 0)
        painter.setPen(QPen(QColor(0, 150, 255), 1))
        painter.drawRect(0, y - window_height // 2, width, window_height)
        painter.end()

        # Scale and display
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

        # Update spectrum plot
        wavelengths = self.spectrometer.pix_to_wave(np.arange(len(profile)))
        self.spectrum_curve.setData(wavelengths, profile)

        # Update peaks if enabled
        show_peaks = getattr(self.spectrometer, 'show_peaks', False)
        if show_peaks:
            peaks = self.spectrometer.find_peaks(profile, height=15, distance=10)
            self.spectrum_peaks.setData(
                wavelengths[peaks],
                profile[peaks],
                symbol='o'
            )
            self.spectrum_peaks.setVisible(True)
        else:
            self.spectrum_peaks.setVisible(False)

        # Update reflectance plot
        if getattr(self.spectrometer, 'reference_spectrum', None) is not None:
            reflectance = self.spectrometer.calculate_reflectance()
            self.reflectance_curve.setData(wavelengths, reflectance)

        # Update colorbar gradient
        if getattr(self.spectrometer, 'show_colorbar', False):
            # Create gradient from violet to red
            colors = [
                (148, 0, 211),  # Violet
                (75, 0, 130),  # Indigo
                (0, 0, 255),  # Blue
                (0, 255, 0),  # Green
                (255, 255, 0),  # Yellow
                (255, 127, 0),  # Orange
                (255, 0, 0)  # Red
            ]
            positions = np.linspace(400, 700, len(colors))
            pos_normalized = (positions - 400) / 300

            self.colorbar.setGradient(pg.ColorMap(pos_normalized, colors))
            self.colorbar.setLabels({0: '400nm', 1: '700nm'})


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
