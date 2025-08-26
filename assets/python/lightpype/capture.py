import pygame
import numpy as np
import cv2
import sys
import os
import json
import matplotlib

# Use non-interactive backend for rendering plots without a GUI window
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from datetime import datetime
import time
from PIL import Image


class Spectrometer:
    """
    Core class handling camera acquisition, calibration,
    HDR processing and spectral data manipulation.
    """
    def __init__(self, camera_index=0):
        # Initialize default settings for the spectrometer
        self.settings = {
            'exposure': -1,  # -1 = automatic exposure
            'gain': 0,
            'roi': (0, 0, 1280, 720),  # Region of interest in the captured frame
            'calibration': {'wavelength': [], 'pixel': []},
            'hdr_exposures': [10, 30, 60, 100],  # Exposure times for HDR capture
            'sensitivity_line': 240,
            'window_height': 10,
            'clipping_threshold': 240,
            'dark_frame': None,
            'reference': None
        }

        self.running = True               # Controls live acquisition loop
        self.calibration_mode = False     # Flag for interactive calibration
        self.hdr_mode = False             # HDR capture mode flag
        self.current_frame = None         # Latest raw frame from camera
        self.profile = None               # Extracted intensity profile (1D)
        self.wavelengths = None           # Wavelength axis after calibration
        self.processed = None              # Profile after dark/reference correction
        self.reflectance = None           # Reflectance calculated from reference
        self.calibration_points = []      # List of user‑added calibration points
        self.capture_start_time = None    # Timestamp for HDR capture start
        self.available_cameras = []       # Detected camera indices

        # Initialize the camera based on provided index
        self.camera_index = camera_index
        self.camera = None
        self.open_camera(camera_index)

        # Attempt to load previously saved configuration; fallback to defaults
        try:
            self.load_config()
        except:
            print("Using default settings")

    def detect_cameras(self):
        """Detect available cameras by testing indices 0‑10."""
        cameras = []
        for i in range(11):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret:
                        cameras.append(i)   # Camera responded with a frame
                except:
                    pass
                finally:
                    cap.release()
        return cameras

    def open_camera(self, index):
        """Open a camera by its index."""
        if self.camera:
            self.camera.release()  # Release previously opened camera

        # Create new VideoCapture object
        self.camera = cv2.VideoCapture(index)
        if not self.camera.isOpened():
            print(f"Failed to open camera index {index}")
            return False

        # Attempt to set a default resolution (1280x720)
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            print("Warning: Could not set default frame size")

        # Update ROI settings to match the chosen resolution
        self.settings['roi'] = (0, 0, 1280, 720)
        return True

    def load_config(self):
        """Load configuration from JSON file if it exists."""
        if os.path.exists("spectrometer_config.json"):
            with open("spectrometer_config.json", 'r') as f:
                self.settings = json.load(f)

    def save_config(self):
        """Persist current settings to a JSON file."""
        with open("spectrometer_config.json", 'w') as f:
            json.dump(self.settings, f, indent=4)

    def set_exposure(self, exposure):
        """Set camera exposure (if supported)."""
        if self.camera and self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.settings['exposure'] = exposure

    def capture_frame(self):
        """
        Capture a single frame from the camera.
        Returns a blank placeholder image when no camera is available.
        """
        if not self.camera or not self.camera.isOpened():
            # Demo mode: generate black image with text
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera", (50, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            return frame

        ret, frame = self.camera.read()
        if ret:
            self.current_frame = frame
        return self.current_frame

    def extract_profile(self, frame, sensitivity_line=None, window_height=None):
        """
        Extract a 1‑D intensity profile from a region of interest.
        Parameters allow overriding the default sensitivity line and window height.
        """
        if frame is None:
            return None

        # Determine line (vertical position) and averaging window size
        line = int(sensitivity_line or self.settings['sensitivity_line'])
        height = int(window_height or self.settings['window_height'])

        # Extract ROI based on settings
        roi = self.settings['roi']
        x, y, w, h = roi

        # Clamp ROI dimensions to frame bounds
        h = min(h, frame.shape[0] - y)
        w = min(w, frame.shape[1] - x)

        region = frame[y:y + h, x:x + w]

        # Convert region to grayscale for intensity analysis
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Compute averaging window around the sensitivity line
        start_line = max(0, line - height // 2)
        end_line = min(gray.shape[0], line + height // 2 + 1)

        # Ensure slice indices are integers
        start_line = int(start_line)
        end_line = int(end_line)

        # Average pixel values across the window to produce a profile
        profile = np.mean(gray[start_line:end_line, :], axis=0)

        self.profile = profile
        return profile

    def process_profile(self, profile):
        """
        Apply dark frame subtraction, reference normalization,
        and wavelength calibration to the raw profile.
        Returns calibrated wavelengths and processed intensity values.
        """
        if profile is None:
            # Return empty arrays when no data is available
            return np.arange(0), np.zeros(0)

        # Dark frame correction (subtract background)
        if self.settings['dark_frame'] is not None:
            profile = profile - self.settings['dark_frame']
            profile = np.clip(profile, 0, None)  # Prevent negative values

        # Reference normalization (divide by reference spectrum)
        if self.settings['reference'] is not None:
            ref = self.settings['reference']
            profile = profile / ref
            profile = np.clip(profile, 0, 1)   # Clamp to [0, 1] range

        self.processed = profile

        # Wavelength calibration using stored pixel‑wavelength pairs
        if self.settings['calibration']['pixel'] and self.settings['calibration']['wavelength']:
            calib_pixels = self.settings['calibration']['pixel']
            calib_wavelengths = self.settings['calibration']['wavelength']
            interp_func = interp1d(calib_pixels, calib_wavelengths,
                                   kind='linear', fill_value='extrapolate')
            self.wavelengths = interp_func(np.arange(len(profile)))
        else:
            # Fallback: use pixel index as wavelength placeholder
            self.wavelengths = np.arange(len(profile))

        return self.wavelengths, profile

    def capture_hdr(self):
        """
        Perform High Dynamic Range (HDR) acquisition.
        If HDR mode is disabled, simply returns a single-profile extraction.
        """
        if not self.hdr_mode:
            # Non‑HDR: extract profile from a single frame
            return self.extract_profile(self.capture_frame())

        profiles = []
        exposures = self.settings['hdr_exposures']
        self.capture_start_time = time.time()  # Record start for progress display

        for exp in exposures:
            self.set_exposure(exp)               # Set camera exposure
            frame = self.capture_frame()
            profile = self.extract_profile(frame)
            profiles.append(profile)

        # Weight each profile by its relative exposure duration
        weights = np.array(exposures) / max(exposures)
        combined = np.zeros_like(profiles[0])

        for i, profile in enumerate(profiles):
            combined += profile * weights[i]

        # Average the weighted sum to produce final HDR profile
        return combined / len(profiles)

    def capture_dark_frame(self):
        """Capture a dark frame (no illumination) and store it as calibration."""
        frame = self.capture_frame()
        profile = self.extract_profile(frame)
        self.settings['dark_frame'] = profile
        self.save_config()

    def capture_reference(self):
        """Capture a reference spectrum for reflectance calculations."""
        frame = self.capture_frame()
        profile = self.extract_profile(frame)
        self.settings['reference'] = profile
        self.save_config()

    def calculate_reflectance(self, sample_profile):
        """
        Compute reflectance by dividing the processed sample spectrum
        by the processed reference spectrum.
        """
        if self.settings['reference'] is None:
            return None

        # Process both reference and sample profiles
        _, ref_processed = self.process_profile(self.settings['reference'])
        _, sample_processed = self.process_profile(sample_profile)

        # Avoid division by zero by enforcing a minimum value
        ref_processed = np.maximum(ref_processed, 1e-6)
        reflectance = sample_processed / ref_processed

        self.reflectance = reflectance
        return reflectance

    def detect_peaks(self, profile, min_height=10, distance=10):
        """Detect spectral peaks using SciPy's find_peaks."""
        peaks, _ = find_peaks(profile, height=min_height, distance=distance)
        return peaks

    def add_calibration_point(self, pixel, wavelength):
        """
        Add a new calibration point (pixel ↔ wavelength) to the stored list.
        Persists changes to configuration file.
        """
        self.calibration_points.append((pixel, wavelength))
        self.settings['calibration']['pixel'].append(pixel)
        self.settings['calibration']['wavelength'].append(wavelength)
        self.save_config()

    def clear_calibration(self):
        """Reset all calibration data."""
        self.calibration_points = []
        self.settings['calibration'] = {'wavelength': [], 'pixel': []}
        self.save_config()


class SpectrometerGUI:
    """
    Pygame‑based graphical user interface for the spectrometer.
    Handles UI elements, camera interaction, and real‑time plotting.
    """
    def __init__(self):
        pygame.init()
        # Initialize spectrometer with an invalid index; will be set later
        self.spec = Spectrometer(-1)
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Spectrometer")

        # Font definitions for UI text
        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 16)

        # Define color palette used throughout the GUI
        self.BG_COLOR = (30, 30, 40)          # Background
        self.PANEL_COLOR = (50, 50, 60)       # Panels
        self.BUTTON_COLOR = (70, 130, 180)    # Buttons
        self.BUTTON_HOVER = (100, 160, 210)   # Hovered button
        self.TEXT_COLOR = (220, 220, 220)     # Text
        self.GRID_COLOR = (80, 80, 90)       # Grid lines

        # UI element rectangles for buttons, checkboxes, sliders
        # Updated positions to add spacing between elements
        self.buttons = {
            'play': pygame.Rect(20, 20, 100, 40),
            'hdr': pygame.Rect(140, 20, 100, 40),          # 20px gap from previous
            'dark': pygame.Rect(260, 20, 100, 40),         # 20px gap
            'reference': pygame.Rect(380, 20, 100, 40),    # 20px gap
            'save': pygame.Rect(500, 20, 100, 40),        # 20px gap
            'calibrate': pygame.Rect(620, 20, 100, 40),   # 20px gap
            'clear_calib': pygame.Rect(740, 20, 100, 40), # 20px gap
            'refresh_cam': pygame.Rect(20, 200, 150, 40),
            'select_cam': pygame.Rect(20, 260, 150, 40),   # moved down to avoid overlap with refresh button
        }

        self.checkboxes = {
            'hdr': pygame.Rect(140, 70, 20, 20),
            'calibration_mode': pygame.Rect(620, 70, 20, 20),
        }

        self.sliders = {
            # Shift sliders down to give more vertical space
            'sensitivity': {'rect': pygame.Rect(20, 120, 200, 20), 'value': 240,
                           'min': 0, 'max': 720},
            'window': {'rect': pygame.Rect(20, 160, 200, 20), 'value': 10,
                       'min': 1, 'max': 100},
        }

        # Plot surfaces will hold rendered Matplotlib images
        self.camera_surface = None
        self.spectrum_surface = None
        self.reflectance_surface = None

        # Detect available cameras and initialise the first working one
        self.camera_list = self.detect_cameras()
        self.selected_camera = 0
        self.init_cameras()

    def detect_cameras(self):
        """Detect available cameras by testing indices 0-10"""
        cameras = []
        # Try default camera first
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened() and cap.read()[0]:
                cameras.append(0)
            cap.release()
        except:
            pass

        # Try additional indices
        for i in range(1, 11):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened() and cap.read()[0]:
                    cameras.append(i)
                cap.release()
            except:
                continue
        return cameras or [0]  # Always include index 0 as fallback

    def init_cameras(self):
        """Initialize available cameras"""
        # Try each detected camera
        for cam_idx in self.camera_list:
            try:
                # Try to open camera
                self.spec.open_camera(cam_idx)
                if self.spec.camera.isOpened():
                    self.selected_camera = cam_idx
                    print(f"Using camera index: {cam_idx}")
                    return
            except Exception as e:
                print(f"Error initializing camera {cam_idx}: {e}")

        print("No working cameras found, using fallback mode")

    def draw_button(self, rect, text, hover=False):
        color = self.BUTTON_HOVER if hover else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        pygame.draw.rect(self.screen, (40, 90, 140), rect, 2, border_radius=5)

        text_surf = self.font.render(text, True, self.TEXT_COLOR)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def draw_checkbox(self, rect, checked, label):
        pygame.draw.rect(self.screen, (60, 60, 80), rect, border_radius=3)
        if checked:
            pygame.draw.rect(self.screen, (100, 200, 100), rect.inflate(-4, -4), border_radius=2)

        label_surf = self.small_font.render(label, True, self.TEXT_COLOR)
        self.screen.blit(label_surf, (rect.x + 30, rect.y))

    def draw_slider(self, slider):
        # Draw track
        pygame.draw.rect(self.screen, (80, 80, 100), slider['rect'])

        # Calculate handle position
        handle_x = slider['rect'].x + int((slider['value'] - slider['min']) /
                                          (slider['max'] - slider['min']) * slider['rect'].width)
        handle_rect = pygame.Rect(handle_x - 10, slider['rect'].y - 5, 20, 30)

        # Draw handle
        pygame.draw.rect(self.screen, (100, 150, 200), handle_rect, border_radius=5)

        # Draw label
        value_text = f"{slider['value']}"
        text_surf = self.small_font.render(value_text, True, self.TEXT_COLOR)
        self.screen.blit(text_surf, (slider['rect'].x, slider['rect'].y - 25))

    def draw_progress_bar(self, duration):
        if not self.spec.hdr_mode or not self.spec.capture_start_time:
            return

        elapsed = time.time() - self.spec.capture_start_time
        progress = min(elapsed / duration, 1.0)

        bar_rect = pygame.Rect(20, 180, 400, 20)
        pygame.draw.rect(self.screen, (80, 80, 100), bar_rect)

        fill_width = int(progress * bar_rect.width)
        fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_width, bar_rect.height)
        pygame.draw.rect(self.screen, (70, 180, 70), fill_rect)

        text = f"HDR Capture: {progress * 100:.1f}%"
        text_surf = self.small_font.render(text, True, self.TEXT_COLOR)
        self.screen.blit(text_surf, (bar_rect.x, bar_rect.y - 25))

    def update_camera_display(self, frame):
        if frame is None:
            # Create blank frame for demo mode
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera", (50, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Resize frame to fit display area
        display_rect = pygame.Rect(450, 100, 700, 400)
        frame = cv2.resize(frame, (display_rect.width, display_rect.height))

        # Convert to RGB and create surface
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.camera_surface = pygame.surfarray.make_surface(rgb_frame.transpose([1, 0, 2]))

        # Draw sensitivity line
        roi_height = self.spec.settings['roi'][3] or 720
        line_pos = int(self.sliders['sensitivity']['value'] * display_rect.height / roi_height)
        line_pos = max(0, min(display_rect.height, line_pos))

        window_height = int(self.sliders['window']['value'])

        # Draw ROI rectangle
        pygame.draw.rect(self.screen, (0, 255, 0), display_rect, 2)

        # Draw sensitivity line
        pygame.draw.line(
            self.camera_surface,
            (255, 0, 0),
            (0, line_pos),
            (display_rect.width, line_pos),
            2
        )

        # Draw window area
        start_y = max(0, line_pos - window_height // 2)
        end_y = min(frame.shape[0], line_pos + window_height // 2)
        pygame.draw.rect(
            self.camera_surface,
            (0, 150, 255),
            (0, int(start_y), display_rect.width, int(end_y - start_y)),
            1
        )

        self.screen.blit(self.camera_surface, display_rect.topleft)

    def update_spectrum_plot(self, wavelengths, spectrum):
        fig = plt.figure(figsize=(6, 3), dpi=80)
        ax = fig.add_subplot(111)

        ax.plot(wavelengths, spectrum, 'b-', linewidth=1.5)
        ax.set_title('Spectrum')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.grid(True, color='0.9')

        # Highlight calibration points
        if self.spec.calibration_points:
            pixels, wavelengths = zip(*self.spec.calibration_points)
            ax.plot(wavelengths, np.interp(wavelengths, self.spec.wavelengths, spectrum),
                    'ro', markersize=6)

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()

        # Correct rendering for matplotlib versions
        try:
            # Older matplotlib versions
            raw_data = renderer.tostring_rgb()
            size = (int(renderer.width), int(renderer.height))
        except AttributeError:
            # Newer versions
            raw_data = canvas.buffer_rgba()
            width = int(renderer.width)
            height = int(renderer.height)
            size = (width, height)
            # Convert RGBA to RGB using PIL
            img = Image.frombytes('RGBA', size, raw_data)
            img = img.convert('RGB')
            raw_data = img.tobytes()

        self.spectrum_surface = pygame.image.fromstring(raw_data, size, "RGB")
        plt.close(fig)

        self.screen.blit(self.spectrum_surface, (450, 520))

    def update_reflectance_plot(self, wavelengths, reflectance):
        fig = plt.figure(figsize=(6, 3), dpi=80)
        ax = fig.add_subplot(111)

        ax.plot(wavelengths, reflectance, 'g-', linewidth=1.5)
        ax.set_title('Reflectance')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_ylim(0, 1.2)
        ax.grid(True, color='0.9')

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()

        # Correct rendering for matplotlib versions
        try:
            # Older matplotlib versions
            raw_data = renderer.tostring_rgb()
            size = (int(renderer.width), int(renderer.height))
        except AttributeError:
            # Newer versions
            raw_data = canvas.buffer_rgba()
            width = int(renderer.width)
            height = int(renderer.height)
            size = (width, height)
            # Convert RGBA to RGB using PIL
            img = Image.frombytes('RGBA', size, raw_data)
            img = img.convert('RGB')
            raw_data = img.tobytes()

        self.reflectance_surface = pygame.image.fromstring(raw_data, size, "RGB")
        plt.close(fig)

        self.screen.blit(self.reflectance_surface, (850, 520))

    def save_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spectrum_{timestamp}.csv"

        if self.spec.wavelengths is not None and self.spec.processed is not None:
            data = np.column_stack((self.spec.wavelengths, self.spec.processed))
            if self.spec.reflectance is not None:
                data = np.column_stack((data, self.spec.reflectance))

            header = "Wavelength(nm),Intensity"
            if self.spec.reflectance is not None:
                header += ",Reflectance"

            np.savetxt(filename, data, delimiter=",", header=header)
            return filename
        return None

    def run(self):
        clock = pygame.time.Clock()
        mouse_down = False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_down = True
                    # Check buttons
                    for name, rect in self.buttons.items():
                        if rect.collidepoint(event.pos):
                            if name == 'play':
                                self.spec.running = not self.spec.running
                            elif name == 'hdr':
                                self.spec.hdr_mode = not self.spec.hdr_mode
                            elif name == 'dark':
                                self.spec.capture_dark_frame()
                            elif name == 'reference':
                                self.spec.capture_reference()
                            elif name == 'save':
                                saved = self.save_data()
                                if saved:
                                    print(f"Saved to {saved}")
                            elif name == 'calibrate':
                                self.spec.calibration_mode = not self.spec.calibration_mode
                            elif name == 'clear_calib':
                                self.spec.clear_calibration()
                            elif name == 'refresh_cam':
                                self.camera_list = self.detect_cameras()
                                if self.camera_list:
                                    self.selected_camera = self.camera_list[0]
                            elif name == 'select_cam':
                                if self.camera_list:
                                    # Cycle through available cameras
                                    current_idx = self.camera_list.index(self.selected_camera)
                                    next_idx = (current_idx + 1) % len(self.camera_list)
                                    self.selected_camera = self.camera_list[next_idx]
                                    self.spec.open_camera(self.selected_camera)

                    # Check checkboxes
                    for name, rect in self.checkboxes.items():
                        if rect.collidepoint(event.pos):
                            if name == 'hdr':
                                self.spec.hdr_mode = not self.spec.hdr_mode
                            elif name == 'calibration_mode':
                                self.spec.calibration_mode = not self.spec.calibration_mode

                    # Check sliders
                    for name, slider in self.sliders.items():
                        if slider['rect'].collidepoint(event.pos):
                            # Update slider value
                            rel_x = event.pos[0] - slider['rect'].x
                            fraction = rel_x / slider['rect'].width
                            value = slider['min'] + fraction * (slider['max'] - slider['min'])
                            self.sliders[name]['value'] = max(slider['min'], min(slider['max'], int(value)))

                            # Update spectrometer setting
                            if name == 'sensitivity':
                                self.spec.settings['sensitivity_line'] = int(value)
                            elif name == 'window':
                                self.spec.settings['window_height'] = int(value)
                            self.spec.save_config()

                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_down = False

                    # Add calibration point
                    if self.spec.calibration_mode and event.button == 3:  # Right click
                        if self.spectrum_surface and self.spec.wavelengths is not None:
                            plot_rect = pygame.Rect(450, 520, 350, 250)
                            if plot_rect.collidepoint(event.pos):
                                rel_x = event.pos[0] - plot_rect.x
                                fraction = rel_x / plot_rect.width
                                pixel = int(fraction * len(self.spec.wavelengths))
                                wavelength = self.spec.wavelengths[pixel]
                                self.spec.add_calibration_point(pixel, wavelength)
                                print(f"Added calibration: pixel={pixel}, wavelength={wavelength:.1f}nm")

            # Update display
            self.screen.fill(self.BG_COLOR)

            # Draw UI panels
            pygame.draw.rect(self.screen, self.PANEL_COLOR, (10, 10, 430, 780))
            pygame.draw.rect(self.screen, self.PANEL_COLOR, (440, 90, 740, 660))

            # Draw buttons
            mouse_pos = pygame.mouse.get_pos()
            for name, rect in self.buttons.items():
                hover = rect.collidepoint(mouse_pos)
                text = name.capitalize()
                if name == 'play':
                    text = "Pause" if self.spec.running else "Play"
                elif name == 'select_cam':
                    text = f"Cam: {self.selected_camera}"
                self.draw_button(rect, text, hover)

            # Draw checkboxes
            for name, rect in self.checkboxes.items():
                checked = getattr(self.spec, name) if hasattr(self.spec, name) else False
                label = name.capitalize().replace('_', ' ')
                self.draw_checkbox(rect, checked, label)

            # Draw sliders
            for name, slider in self.sliders.items():
                label = name.capitalize()
                label_surf = self.small_font.render(label, True, self.TEXT_COLOR)
                self.screen.blit(label_surf, (slider['rect'].x, slider['rect'].y - 25))
                self.draw_slider(slider)

            # Draw camera list info
            cam_text = f"Available cameras: {self.camera_list}"
            cam_surf = self.small_font.render(cam_text, True, self.TEXT_COLOR)
            self.screen.blit(cam_surf, (20, 160))

            # Current camera info
            cam_text = f"Selected camera: {self.selected_camera}"
            cam_surf = self.small_font.render(cam_text, True, self.TEXT_COLOR)
            self.screen.blit(cam_surf, (20, 180))

            # Capture and process frame
            if self.spec.running:
                if self.spec.hdr_mode:
                    total_duration = sum(self.spec.settings['hdr_exposures']) / 1000.0
                    self.draw_progress_bar(total_duration)

                    if self.spec.capture_start_time:
                        elapsed = time.time() - self.spec.capture_start_time
                        if elapsed >= total_duration:
                            profile = self.spec.capture_hdr()
                            self.spec.capture_start_time = None

                            # Process and calculate reflectance
                            wavelengths, processed = self.spec.process_profile(profile)
                            if self.spec.settings['reference'] is not None:
                                self.spec.calculate_reflectance(profile)
                else:
                    frame = self.spec.capture_frame()
                    profile = self.spec.extract_profile(
                        frame,
                        self.sliders['sensitivity']['value'],
                        self.sliders['window']['value']
                    )

                    # Process and calculate reflectance
                    wavelengths, processed = self.spec.process_profile(profile)
                    if self.spec.settings['reference'] is not None:
                        self.spec.calculate_reflectance(profile)

                    # Update displays
                    self.update_camera_display(frame)
                    if wavelengths is not None and processed is not None:
                        self.update_spectrum_plot(wavelengths, processed)
                    if self.spec.reflectance is not None:
                        self.update_reflectance_plot(wavelengths, self.spec.reflectance)

            # Update display
            pygame.display.flip()
            clock.tick(30)


if __name__ == "__main__":
    try:
        gui = SpectrometerGUI()
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()
        sys.exit()
