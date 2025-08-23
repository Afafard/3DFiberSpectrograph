import pygame
import numpy as np
import cv2
import sys
import os
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from datetime import datetime


class Spectrometer:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError("Cannot open camera")

        # Default settings
        self.settings = {
            'exposure': -1,  # -1 = automatic
            'gain': 0,
            'roi': (0, 0, 1280, 720),
            'calibration': {'wavelength': [], 'pixel': []},
            'hdr_exposures': [10, 30, 60, 100],
            'sensitivity_line': 240,
            'window_height': 10,
            'clipping_threshold': 240,
            'dark_frame': None,
            'reference': None
        }

        self.running = True
        self.calibration_mode = False
        self.hdr_mode = False
        self.current_frame = None
        self.profile = None
        self.wavelengths = None
        self.processed = None
        self.reflectance = None
        self.calibration_points = []
        self.capture_start_time = None

        # Try to load config
        try:
            self.load_config()
        except:
            print("Using default settings")

    def load_config(self):
        if os.path.exists("spectrometer_config.json"):
            with open("spectrometer_config.json", 'r') as f:
                self.settings = json.load(f)

    def save_config(self):
        with open("spectrometer_config.json", 'w') as f:
            json.dump(self.settings, f, indent=4)

    def set_exposure(self, exposure):
        self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.settings['exposure'] = exposure

    def capture_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.current_frame = frame
        return self.current_frame

    def extract_profile(self, frame, sensitivity_line=None, window_height=None):
        line = sensitivity_line or self.settings['sensitivity_line']
        height = window_height or self.settings['window_height']

        # Extract region of interest
        roi = self.settings['roi']
        x, y, w, h = roi
        region = frame[y:y + h, x:x + w]

        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Extract profile with averaging
        start_line = max(0, line - height // 2)
        end_line = min(gray.shape[0], line + height // 2 + 1)
        profile = np.mean(gray[start_line:end_line, :], axis=0)

        self.profile = profile
        return profile

    def process_profile(self, profile):
        # Apply dark frame correction
        if self.settings['dark_frame'] is not None:
            profile = profile - self.settings['dark_frame']
            profile = np.clip(profile, 0, None)

        # Apply intensity calibration
        if self.settings['reference'] is not None:
            ref = self.settings['reference']
            profile = profile / ref
            profile = np.clip(profile, 0, 1)

        self.processed = profile

        # Apply wavelength calibration if available
        if self.settings['calibration']['pixel'] and self.settings['calibration']['wavelength']:
            calib_pixels = self.settings['calibration']['pixel']
            calib_wavelengths = self.settings['calibration']['wavelength']
            interp_func = interp1d(calib_pixels, calib_wavelengths,
                                   kind='linear', fill_value='extrapolate')
            self.wavelengths = interp_func(np.arange(len(profile)))
        else:
            self.wavelengths = np.arange(len(profile))

        return self.wavelengths, profile

    def capture_hdr(self):
        if not self.hdr_mode:
            return self.extract_profile(self.capture_frame())

        profiles = []
        exposures = self.settings['hdr_exposures']
        self.capture_start_time = datetime.now()

        for exp in exposures:
            self.set_exposure(exp)
            frame = self.capture_frame()
            profile = self.extract_profile(frame)
            profiles.append(profile)

        # Weight by exposure time
        weights = np.array(exposures) / max(exposures)
        combined = np.zeros_like(profiles[0])

        for i, profile in enumerate(profiles):
            combined += profile * weights[i]

        return combined / len(profiles)

    def capture_dark_frame(self):
        frame = self.capture_frame()
        profile = self.extract_profile(frame)
        self.settings['dark_frame'] = profile
        self.save_config()

    def capture_reference(self):
        frame = self.capture_frame()
        profile = self.extract_profile(frame)
        self.settings['reference'] = profile
        self.save_config()

    def calculate_reflectance(self, sample_profile):
        if self.settings['reference'] is None:
            return None

        # Process both profiles
        _, ref_processed = self.process_profile(self.settings['reference'])
        _, sample_processed = self.process_profile(sample_profile)

        # Avoid division by zero
        ref_processed = np.maximum(ref_processed, 1e-6)
        reflectance = sample_processed / ref_processed

        self.reflectance = reflectance
        return reflectance

    def detect_peaks(self, profile, min_height=10, distance=10):
        peaks, _ = find_peaks(profile, height=min_height, distance=distance)
        return peaks

    def add_calibration_point(self, pixel, wavelength):
        self.calibration_points.append((pixel, wavelength))
        self.settings['calibration']['pixel'].append(pixel)
        self.settings['calibration']['wavelength'].append(wavelength)
        self.save_config()

    def clear_calibration(self):
        self.calibration_points = []
        self.settings['calibration'] = {'wavelength': [], 'pixel': []}
        self.save_config()


class SpectrometerGUI:
    def __init__(self, spectrometer):
        pygame.init()
        self.spec = spectrometer
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Spectrometer")

        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 16)

        # Colors
        self.BG_COLOR = (30, 30, 40)
        self.PANEL_COLOR = (50, 50, 60)
        self.BUTTON_COLOR = (70, 130, 180)
        self.BUTTON_HOVER = (100, 160, 210)
        self.TEXT_COLOR = (220, 220, 220)
        self.GRID_COLOR = (80, 80, 90)

        # UI elements
        self.buttons = {
            'play': pygame.Rect(20, 20, 100, 40),
            'hdr': pygame.Rect(140, 20, 100, 40),
            'dark': pygame.Rect(260, 20, 100, 40),
            'reference': pygame.Rect(380, 20, 100, 40),
            'save': pygame.Rect(500, 20, 100, 40),
            'calibrate': pygame.Rect(620, 20, 100, 40),
            'clear_calib': pygame.Rect(740, 20, 100, 40),
        }

        self.checkboxes = {
            'hdr': pygame.Rect(140, 70, 20, 20),
            'calibration_mode': pygame.Rect(620, 70, 20, 20),
        }

        self.sliders = {
            'sensitivity': {'rect': pygame.Rect(20, 100, 200, 20), 'value': self.spec.settings['sensitivity_line'],
                            'min': 0, 'max': self.spec.settings['roi'][3]},
            'window': {'rect': pygame.Rect(20, 140, 200, 20), 'value': self.spec.settings['window_height'], 'min': 1,
                       'max': 100},
        }

        # Plot surfaces
        self.camera_surface = None
        self.spectrum_surface = None
        self.reflectance_surface = None

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

        elapsed = (datetime.now() - self.spec.capture_start_time).total_seconds()
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
            return

        # Resize frame to fit display area
        display_rect = pygame.Rect(450, 100, 700, 400)
        frame = cv2.resize(frame, (display_rect.width, display_rect.height))

        # Convert to RGB and create surface
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.camera_surface = pygame.surfarray.make_surface(rgb_frame.transpose([1, 0, 2]))

        # Draw sensitivity line
        line_pos = int(self.sliders['sensitivity']['value'] * display_rect.height / self.spec.settings['roi'][3])
        window_height = self.sliders['window']['value']

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
            (0, start_y, display_rect.width, end_y - start_y),
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
        raw_data = renderer.tostring_rgb()

        size = canvas.get_width_height()
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
        raw_data = renderer.tostring_rgb()

        size = canvas.get_width_height()
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

            # Capture and process frame
            if self.spec.running:
                if self.spec.hdr_mode:
                    total_duration = sum(self.spec.settings['hdr_exposures']) / 1000.0
                    self.draw_progress_bar(total_duration)

                    if self.spec.capture_start_time:
                        elapsed = (datetime.now() - self.spec.capture_start_time).total_seconds()
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
                    self.update_spectrum_plot(wavelengths, processed)
                    if self.spec.reflectance is not None:
                        self.update_reflectance_plot(wavelengths, self.spec.reflectance)

            # Update display
            pygame.display.flip()
            clock.tick(30)


if __name__ == "__main__":
    try:
        spec = Spectrometer()
        gui = SpectrometerGUI(spec)
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
        sys.exit()
