import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import cv2
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QFileDialog, QSplitter, 
                             QGroupBox, QFormLayout, QSlider, QSizePolicy, QRadioButton,
                             QFrame, QButtonGroup)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLLinePlotItem, GLMeshItem
from OpenGL.GL import *
from OpenGL.GLU import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# Custom color scheme
GRAPH_BG_COLOR = '#ffffff'
GRAPH_TEXT_COLOR = '#333333'
GRAPH_GRID_COLOR = '#e0e0e0'
GRAPH_LINE_COLOR = '#1f77b4'
GRAPH_HIGHLIGHT_COLOR = '#ff7f0e'

# Available variables by data type
VARIABLES = {
    "FFS": ["X_adc", "Y_adc", "X_load", "Y_load", "Res_Force", "X_Flow", "Y_Flow", 
            "Res_Flow", "Res_Angle", "Temperature"],
    "IMU": ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", 
            "Mag_X", "Mag_Y", "Mag_Z", "Yaw", "Pitch", "Roll"],
    "$GNRMC": ["Speed_Knots", "Angle"]
}

class SurfboardVisualizer(GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create grid
        self.grid = GLGridItem()
        self.grid.scale(2, 2, 1)
        self.addItem(self.grid)

        # Board drawing
        verts = np.array([
            [0.0, 0.0, 0.0],      # Center point
            [-0.8, -0.07, -0.05],   # Tail left 
            [-0.8, 0.07, -0.05],    # Tail right
            [-0.5, -0.3, -0.1],     # Low mid left
            [-0.5, 0.3, -0.1],      # Low mid right
            [0.3, -0.3, -0.1],      # High mid left
            [0.3, 0.3, -0.1],       # High mid right
            [1.0, 0.0, 0.0],        # Nose point
        ])

        # Center the board
        center = verts.mean(axis=0)
        verts -= center

        # Define faces
        faces = np.array([
            [1, 2, 3], [2, 4, 3],    # Tail to Low Mid
            [3, 4, 5], [4, 6, 5],    # Low Mid to High Mid
            [5, 6, 7],               # High Mid to Nose
        ])

        colors = np.array([[0.8, 0.7, 0.6, 1.0] for _ in range(len(faces))])

        self.surfboard = GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=colors,
            smooth=False,
            drawEdges=True,
            edgeColor=(0, 0, 0, 1)
        )
        self.addItem(self.surfboard)

        # Coordinate axes
        axis_x = GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(1, 0, 0, 1), width=2)
        axis_y = GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 1, 0]]), color=(0, 1, 0, 1), width=2)
        axis_z = GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 1]]), color=(0, 0, 1, 1), width=2)

        self.addItem(axis_x)
        self.addItem(axis_y)
        self.addItem(axis_z)

        # Initial camera position
        self.setCameraPosition(distance=3, elevation=20, azimuth=45)

    def update_orientation(self, yaw, pitch, roll):
        """Update board orientation based on IMU data"""
        self.surfboard.resetTransform()
        self.surfboard.rotate(roll, 1, 0, 0, local=True)    # Roll
        self.surfboard.rotate(pitch, 0, 1, 0, local=True)   # Pitch
        self.surfboard.rotate(yaw, 0, 0, 1, local=True)     # Yaw


class VideoDataPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surf Data Visualization Dashboard - Advanced Analysis")
        self.setGeometry(100, 100, 1800, 1000)
        
        # Initialize variables
        self.conn = None
        self.video_capture = None
        self.video_fps = 30
        self.playback_speed = 1.0
        self.is_playing = False
        self.timer = QTimer()
        self.event_data = pd.DataFrame()
        self.imu_data = pd.DataFrame()
        self.gps_data = pd.DataFrame()
        self.cursor_line = None
        self.current_event = None
        self.speed_unit = 'knots'  # Default speed unit
        self.current_frame = 0
        self.colorbar = None
        self.gps_path = None
        self.speed_path = None
        self.paused_time = 0  # Track time when paused
        
        # Configure UI
        self.init_ui()
        self.timer.timeout.connect(self.update_frame)

    def check_checksum(self, sentence):
        """Verify GNRMC checksum"""
        sentence = sentence.strip()
        match = re.match(r'^\$(.*)\*(\w\w)$', sentence)
        if not match:
            return False
        data, checksum = match.groups()
        calculated_checksum = 0
        for char in data:
            calculated_checksum ^= ord(char)
        return f'{calculated_checksum:02X}' == checksum.upper()

    def parse_gnrmc_message(self, message):
        """Enhanced GNRMC parsing with checksum verification"""
        if not self.check_checksum(message):
            return None
            
        parts = message.split(',')
        if len(parts) < 10 or parts[0] != '$GNRMC':
            return None
        
        try:
            # Parse time (HHMMSS.SSS)
            time_str = parts[1]
            
            # Parse latitude (DDMM.MMMM) and direction (N/S)
            lat_str = parts[3]
            lat_dir = parts[4]
            lat_deg = float(lat_str[:2])
            lat_min = float(lat_str[2:])
            latitude = lat_deg + lat_min / 60.0
            if lat_dir == 'S':
                latitude = -latitude
                
            # Parse longitude (DDDMM.MMMM) and direction (E/W)
            lon_str = parts[5]
            lon_dir = parts[6]
            lon_deg = float(lon_str[:3])
            lon_min = float(lon_str[3:])
            longitude = lon_deg + lon_min / 60.0
            if lon_dir == 'W':
                longitude = -longitude
                
            # Parse speed (knots) and angle
            speed = float(parts[7]) if parts[7] else 0.0
            angle = float(parts[8]) if parts[8] else 0.0
            
            return {
                'time': time_str,
                'latitude': latitude,
                'longitude': longitude,
                'speed': speed,
                'angle': angle,
                'valid': parts[2] == 'A'
            }
        except (ValueError, IndexError):
            return None

    def init_ui(self):
        # Main splitter for data and video sections
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # Left panel for data visualization
        data_panel = QWidget()
        data_layout = QVBoxLayout(data_panel)
        
        # Right panel for video and 3D visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add panels to splitter
        main_splitter.addWidget(data_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([1200, 800])
        
        # ===== DATA PANEL =====
        # Control panel group
        control_group = QGroupBox("Data Controls")
        control_layout = QVBoxLayout()
        
        # File import buttons
        file_buttons = QHBoxLayout()
        self.db_button = QPushButton("Load Database")
        self.db_button.clicked.connect(self.load_database)
        file_buttons.addWidget(self.db_button)
        
        self.video_button = QPushButton("Load Video")
        self.video_button.clicked.connect(self.load_video)
        file_buttons.addWidget(self.video_button)
        
        control_layout.addLayout(file_buttons)
        
        # Event selection
        event_form = QFormLayout()
        self.event_selector = QComboBox()
        self.event_selector.setPlaceholderText("Select an event")
        event_form.addRow("Event:", self.event_selector)
        self.event_selector.currentTextChanged.connect(self.prepare_event)
        
        # Sensor selection
        self.type_selector = QComboBox()
        self.type_selector.addItems(VARIABLES.keys())
        self.type_selector.setCurrentText("FFS")  # Set FFS as default
        event_form.addRow("Sensor Type:", self.type_selector)
        self.type_selector.currentTextChanged.connect(self.update_variable_selector)
        
        # Variable selection
        self.variable_selector = QComboBox()
        event_form.addRow("Variable:", self.variable_selector)
        self.variable_selector.currentTextChanged.connect(self.update_plot)
        
        control_layout.addLayout(event_form)
        control_group.setLayout(control_layout)
        data_layout.addWidget(control_group)
        
        # Main data plot
        self.figure, self.ax = plt.subplots(figsize=(12, 4), facecolor=GRAPH_BG_COLOR)
        self.ax.set_facecolor(GRAPH_BG_COLOR)
        self.ax.tick_params(colors=GRAPH_TEXT_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color(GRAPH_TEXT_COLOR)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        data_layout.addWidget(self.toolbar)
        data_layout.addWidget(self.canvas)
        
        # GPS and Speed visualization
        gps_speed_splitter = QSplitter(Qt.Horizontal)
        
        # GPS Map visualization
        self.gps_map_fig, self.gps_map_ax = plt.subplots(
            figsize=(6, 4),
            subplot_kw={'projection': ccrs.PlateCarree()},
            facecolor=GRAPH_BG_COLOR
        )
        self.gps_map_ax.set_facecolor=GRAPH_BG_COLOR
        self.gps_map_ax.add_feature(cfeature.LAND, color='#f0f0f0')
        self.gps_map_ax.add_feature(cfeature.OCEAN, color='#e6f3ff')
        self.gps_map_ax.add_feature(cfeature.COASTLINE, edgecolor=GRAPH_TEXT_COLOR)
        self.gps_map_ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor=GRAPH_TEXT_COLOR)
        self.gps_map_canvas = FigureCanvas(self.gps_map_fig)
        
        # Speed plot
        self.speed_fig, self.speed_ax = plt.subplots(figsize=(6, 4), facecolor=GRAPH_BG_COLOR)
        self.speed_ax.set_facecolor=GRAPH_BG_COLOR
        self.speed_ax.tick_params(colors=GRAPH_TEXT_COLOR)
        for spine in self.speed_ax.spines.values():
            spine.set_color(GRAPH_TEXT_COLOR)
        self.speed_canvas = FigureCanvas(self.speed_fig)
        
        gps_speed_splitter.addWidget(self.gps_map_canvas)
        
        speed_widget = QWidget()
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(self.speed_canvas)
        
        # Speed unit selection (now below the speed plot)
        speed_unit_group = QGroupBox("Speed Units")
        speed_unit_layout = QHBoxLayout()
        self.knots_radio = QRadioButton("Knots")
        self.knots_radio.setChecked(True)
        self.knots_radio.toggled.connect(lambda: self.set_speed_unit('knots'))
        self.kmh_radio = QRadioButton("km/h")
        self.kmh_radio.toggled.connect(lambda: self.set_speed_unit('kmh'))
        self.ms_radio = QRadioButton("m/s")
        self.ms_radio.toggled.connect(lambda: self.set_speed_unit('ms'))
        
        # Add radio buttons to a button group
        self.speed_unit_group = QButtonGroup()
        self.speed_unit_group.addButton(self.knots_radio)
        self.speed_unit_group.addButton(self.kmh_radio)
        self.speed_unit_group.addButton(self.ms_radio)
        
        speed_unit_layout.addWidget(self.knots_radio)
        speed_unit_layout.addWidget(self.kmh_radio)
        speed_unit_layout.addWidget(self.ms_radio)
        speed_unit_group.setLayout(speed_unit_layout)
        
        speed_layout.addWidget(speed_unit_group)
        speed_widget.setLayout(speed_layout)
        
        gps_speed_splitter.addWidget(speed_widget)
        gps_speed_splitter.setSizes([600, 600])
        
        data_layout.addWidget(gps_speed_splitter)
        
        # ===== RIGHT PANEL =====
        # Video control group
        video_control_group = QGroupBox("Video Controls")
        video_control_layout = QVBoxLayout()
        
        # Playback controls - now with separate buttons
        playback_buttons = QHBoxLayout()
        
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.start_playback)
        playback_buttons.addWidget(self.play_button)
        
        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.clicked.connect(self.pause_playback)
        self.pause_button.setEnabled(False)  # Disabled until playback starts
        playback_buttons.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)  # Disabled until playback starts
        playback_buttons.addWidget(self.stop_button)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Playback Speed:"))
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)  # 0.1x to 2.0x in 0.1 increments
        self.speed_slider.setValue(10)     # Default to 1.0x
        self.speed_slider.valueChanged.connect(self.update_playback_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_label)
        
        video_control_layout.addLayout(playback_buttons)
        video_control_layout.addLayout(speed_layout)
        video_control_group.setLayout(video_control_layout)
        right_layout.addWidget(video_control_group)
        
        # Video display
        self.video_frame_label = QLabel()
        self.video_frame_label.setAlignment(Qt.AlignCenter)
        self.video_frame_label.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.video_frame_label, stretch=2)
        
        # 3D Surfboard visualization
        self.surfboard_viz = SurfboardVisualizer()
        right_layout.addWidget(self.surfboard_viz, stretch=1)
        
        # Status information
        self.status_label = QLabel("Ready to load data and video")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)
        
        # Initialize variable selector with default FFS and Res_Flow
        self.update_variable_selector()
        self.variable_selector.setCurrentText("Res_Flow")

    def set_speed_unit(self, unit):
        """Set speed unit and update speed plot"""
        self.speed_unit = unit
        self.update_speed_plot()

    def update_speed_plot(self):
        """Update speed plot with selected units and color by speed"""
        if self.gps_data.empty or 'Timestamp' not in self.gps_data.columns:
            return
            
        self.speed_ax.clear()
        
        # Convert speed based on selected unit
        if self.speed_unit == 'knots':
            speed = self.gps_data['Speed_Knots']
            unit_label = "Speed (knots)"
        elif self.speed_unit == 'kmh':
            speed = self.gps_data['Speed_Knots'] * 1.852
            unit_label = "Speed (km/h)"
        else:  # m/s
            speed = self.gps_data['Speed_Knots'] * 0.514444
            unit_label = "Speed (m/s)"
        
        # Create color segments based on speed
        points = np.array([self.gps_data['Timestamp'], speed]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a colormap for the speed
        norm = plt.Normalize(speed.min(), speed.max())
        cmap = cm.get_cmap('viridis')
        
        # Create line collection
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(speed)
        lc.set_linewidth(2)
        self.speed_path = self.speed_ax.add_collection(lc)
        
        # Add colorbar below the plot
        if self.colorbar:
            self.colorbar.remove()
        self.colorbar = self.speed_fig.colorbar(lc, ax=self.speed_ax, orientation='horizontal', pad=0.2)
        self.colorbar.set_label(unit_label, color=GRAPH_TEXT_COLOR)
        self.colorbar.ax.xaxis.set_tick_params(color=GRAPH_TEXT_COLOR)
        plt.setp(plt.getp(self.colorbar.ax.axes, 'xticklabels'), color=GRAPH_TEXT_COLOR)
        
        # Set axis limits
        self.speed_ax.set_xlim(self.gps_data['Timestamp'].min(), self.gps_data['Timestamp'].max())
        self.speed_ax.set_ylim(speed.min() * 0.95, speed.max() * 1.05)
        
        # Formatting
        self.speed_ax.set_xlabel("Time", color=GRAPH_TEXT_COLOR)
        self.speed_ax.set_ylabel(unit_label, color=GRAPH_TEXT_COLOR)
        self.speed_ax.grid(True, color=GRAPH_GRID_COLOR, linestyle='--', alpha=0.7)
        self.speed_ax.set_title(f"GPS Speed ({unit_label})", color=GRAPH_TEXT_COLOR, pad=10)
        
        # Format x-axis ticks
        plt.setp(self.speed_ax.get_xticklabels(), rotation=45, ha='right', color=GRAPH_TEXT_COLOR)
        plt.setp(self.speed_ax.get_yticklabels(), color=GRAPH_TEXT_COLOR)
        
        self.speed_fig.tight_layout()
        self.speed_canvas.draw()

    def update_gps_map(self):
        """Update GPS tracking map with colored path"""
        if self.gps_data.empty or 'Timestamp' not in self.gps_data.columns:
            return
            
        self.gps_map_ax.clear()
        
        # Add map features
        self.gps_map_ax.add_feature(cfeature.LAND, color='#f0f0f0')
        self.gps_map_ax.add_feature(cfeature.OCEAN, color='#e6f3ff')
        self.gps_map_ax.add_feature(cfeature.COASTLINE, edgecolor=GRAPH_TEXT_COLOR)
        self.gps_map_ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor=GRAPH_TEXT_COLOR)
        
        # Plot GPS track colored by speed
        points = np.array([self.gps_data['Longitude'], self.gps_data['Latitude']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create line collection with color based on speed
        norm = plt.Normalize(self.gps_data['Speed_Knots'].min(), self.gps_data['Speed_Knots'].max())
        cmap = cm.get_cmap('viridis')
        
        lc = LineCollection(segments, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        lc.set_array(self.gps_data['Speed_Knots'])
        lc.set_linewidth(2)
        self.gps_path = self.gps_map_ax.add_collection(lc)
        
        # Add colorbar
        if hasattr(self, 'map_colorbar'):
            self.map_colorbar.remove()
        self.map_colorbar = self.gps_map_fig.colorbar(lc, ax=self.gps_map_ax, orientation='horizontal', pad=0.05)
        self.map_colorbar.set_label('Speed (knots)', color=GRAPH_TEXT_COLOR)
        self.map_colorbar.ax.xaxis.set_tick_params(color=GRAPH_TEXT_COLOR)
        plt.setp(plt.getp(self.map_colorbar.ax.axes, 'xticklabels'), color=GRAPH_TEXT_COLOR)
        
        # Set map extent with padding
        lat_pad = max(0.01, 0.01 * (self.gps_data['Latitude'].max() - self.gps_data['Latitude'].min()))
        lon_pad = max(0.01, 0.01 * (self.gps_data['Longitude'].max() - self.gps_data['Longitude'].min()))
        
        self.gps_map_ax.set_extent([
            self.gps_data['Longitude'].min() - lon_pad,
            self.gps_data['Longitude'].max() + lon_pad,
            self.gps_data['Latitude'].min() - lat_pad,
            self.gps_data['Latitude'].max() + lat_pad
        ], crs=ccrs.PlateCarree())
        
        # Add start and end markers
        self.gps_map_ax.plot(
            self.gps_data['Longitude'].iloc[0],
            self.gps_data['Latitude'].iloc[0],
            'go',
            markersize=8,
            transform=ccrs.PlateCarree(),
            label='Start'
        )
        
        self.gps_map_ax.plot(
            self.gps_data['Longitude'].iloc[-1],
            self.gps_data['Latitude'].iloc[-1],
            'ro',
            markersize=8,
            transform=ccrs.PlateCarree(),
            label='End'
        )
        
        self.gps_map_ax.legend(loc='upper right')
        
        self.gps_map_fig.tight_layout()
        self.gps_map_canvas.draw()

    def update_playback_speed(self):
        """Update playback speed based on slider value"""
        self.playback_speed = self.speed_slider.value() / 10.0
        self.speed_label.setText(f"{self.playback_speed:.1f}x")
        
        if self.is_playing:
            self.timer.stop()
            self.timer.start(int(1000 / (self.video_fps * self.playback_speed)))

    def start_playback(self):
        """Start video playback"""
        if self.video_capture is None:
            self.status_label.setText("Please load video file first")
            return
            
        if self.current_event is None:
            self.status_label.setText("Please select an event first")
            return

        # If we're paused, resume from paused time
        if self.paused_time > 0:
            start_video_seconds = self.paused_time
            self.paused_time = 0
        else:
            start_video_seconds = self.parse_time(self.current_event['Start'])
            
        self.video_capture.set(cv2.CAP_PROP_POS_MSEC, start_video_seconds * 1000)
        self.current_frame = 0
        self.timer.start(int(1000 / (self.video_fps * self.playback_speed)))
        self.is_playing = True
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Playing...")

    def pause_playback(self):
        """Pause video playback and remember position"""
        if not self.is_playing:
            return
            
        # Store current position for resuming
        self.paused_time = self.video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        self.timer.stop()
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Paused")

    def stop_playback(self):
        """Stop video playback and reset"""
        self.timer.stop()
        self.is_playing = False
        self.paused_time = 0
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Playback stopped")
        
        # Reset cursor position if we have event data
        if not self.event_data.empty and self.cursor_line:
            self.cursor_line.set_xdata([self.event_data['Timestamp'].iloc[0]])
            self.canvas.draw()

    def load_database(self):
        """Load SQLite database file and populate events"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SQLite Database", 
            os.getcwd(), "Database Files (*.db)"
        )
        if file_path:
            try:
                self.conn = sqlite3.connect(file_path)
                
                # Load events directly from database
                events = pd.read_sql_query("SELECT Event_Name FROM Events ORDER BY Event_Name", self.conn)
                
                self.event_selector.clear()
                if not events.empty:
                    self.event_selector.addItems(events['Event_Name'].tolist())
                    self.event_selector.setCurrentIndex(0)  # Select first event
                    self.prepare_event()  # Load data for first event
                    self.status_label.setText(f"Database loaded: {os.path.basename(file_path)} with {len(events)} events")
                else:
                    self.status_label.setText(f"Database loaded but no events found")
                
                print(f"Database loaded successfully with {len(events)} events")
            except Exception as e:
                self.status_label.setText(f"Error loading database: {str(e)}")
                print(f"Database load error: {e}")

    def load_video(self):
        """Load video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", 
            os.getcwd(), "Video Files (*.mp4 *.avi)"
        )
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            if self.video_capture.isOpened():
                self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                self.status_label.setText(f"Video loaded: {os.path.basename(file_path)} @ {self.video_fps:.1f} FPS")
                print(f"Video loaded at {self.video_fps} FPS.")
            else:
                self.status_label.setText("Failed to load video file")
                print("Video load failed")

    def update_variable_selector(self):
        """Update available variables based on selected sensor type"""
        sensor_type = self.type_selector.currentText()
        self.variable_selector.clear()
        self.variable_selector.addItems(VARIABLES[sensor_type])

        # Atualiza corretamente o evento filtrado para o sensor selecionado
        if hasattr(self, 'full_event_data'):
            self.event_data = self.full_event_data[self.full_event_data['Data_Type'] == sensor_type]
            print(f"Event data updated with {len(self.event_data)} points for sensor {sensor_type}")

        self.update_plot()

    def prepare_event(self):
        """Prepare data for selected event using database"""
        if self.conn is None:
            self.status_label.setText("Please load database first")
            return

        event_name = self.event_selector.currentText()
        if not event_name:
            return

        print(f"\nPreparing event: {event_name}")

        # Get event details directly from database
        event_df = pd.read_sql_query(
            "SELECT * FROM Events WHERE Event_Name = ?",
            self.conn, params=(event_name,)
        )

        if event_df.empty:
            self.status_label.setText(f"No event found: {event_name}")
            print(f"No event found in database for: {event_name}")
            return

        self.current_event = event_df.iloc[0]
        print(f"Event details: {self.current_event}")

        # Load all sensor data for this event
        df = pd.read_sql_query(
            "SELECT * FROM Sensor_Data WHERE Event_ID = ? ORDER BY Time, Millisec",
            self.conn, params=(self.current_event['Event_ID'],)
        )

        if df.empty:
            self.status_label.setText(f"No data found for event: {event_name}")
            print(f"No sensor data found for event: {event_name}")
            self.event_data = pd.DataFrame()
            self.imu_data = pd.DataFrame()
            self.gps_data = pd.DataFrame()
            return

        print(f"Found {len(df)} raw data points")
        print("Sample data:")
        print(df.head())

        # Process timestamps - critical fix for database format
        try:
            # Handle milliseconds properly - they're stored as integers in the database
            df['Millisec'] = df['Millisec'].fillna(0).astype(int).astype(str).str.zfill(3)
            
            # Combine time and milliseconds
            df['Timestamp'] = pd.to_datetime(
                df['Time'] + '.' + df['Millisec'],
                format='%H:%M:%S.%f',
                errors='coerce'
            )
            
            # Drop rows with invalid timestamps
            df = df.dropna(subset=['Timestamp'])
            
            if df.empty:
                raise ValueError("No valid timestamps after processing")
                
            print(f"Timestamps processed successfully, {len(df)} valid points")
        except Exception as e:
            self.status_label.setText(f"Error processing timestamps: {str(e)}")
            print(f"Timestamp processing error: {e}")
            return

        # Store all data for later filtering
        self.full_event_data = df
        
        # Filter data by current sensor type
        self.event_data = df[df['Data_Type'] == self.type_selector.currentText()]
        
        # Store IMU data separately for surfboard visualization
        self.imu_data = df[df['Data_Type'] == "IMU"].copy()
        if not self.imu_data.empty:
            print(f"Found {len(self.imu_data)} IMU data points")
            try:
                # Parse IMU data columns - handle both comma-separated and space-separated formats
                if self.imu_data['Data'].str.contains(',').all():
                    imu_values = self.imu_data['Data'].str.split(',', expand=True)
                else:
                    imu_values = self.imu_data['Data'].str.split(expand=True)
                
                # Convert to numeric, handling errors
                imu_values = imu_values.apply(pd.to_numeric, errors='coerce')
                
                # Assign to DataFrame
                self.imu_data['Yaw'] = imu_values.iloc[:, VARIABLES["IMU"].index("Yaw")]
                self.imu_data['Pitch'] = imu_values.iloc[:, VARIABLES["IMU"].index("Pitch")]
                self.imu_data['Roll'] = imu_values.iloc[:, VARIABLES["IMU"].index("Roll")]
                print("IMU data parsed successfully")
            except Exception as e:
                print(f"Error parsing IMU data: {e}")

        # Store GPS data separately for map visualization
        self.gps_data = df[df['Data_Type'] == "$GNRMC"].copy()
        if not self.gps_data.empty:
            print(f"Found {len(self.gps_data)} GPS data points")
            # Parse each GPS message
            gps_records = []
            for _, row in self.gps_data.iterrows():
                # The GNRMC message is stored directly in the Data column
                gnrmc_message = row['Data'].strip()
                parsed = self.parse_gnrmc_message(gnrmc_message)
                if parsed and parsed['valid']:
                    gps_records.append({
                        'Timestamp': row['Timestamp'],
                        'Latitude': parsed['latitude'],
                        'Longitude': parsed['longitude'],
                        'Speed_Knots': parsed['speed'],
                        'Angle': parsed['angle']
                    })
            
            if gps_records:
                self.gps_data = pd.DataFrame(gps_records)
                print(f"Parsed {len(self.gps_data)} valid GPS records")
                
                # Calculate additional speed metrics
                self.gps_data['Speed_kmh'] = self.gps_data['Speed_Knots'] * 1.852
                self.gps_data['Speed_ms'] = self.gps_data['Speed_Knots'] * 0.514444
                
                # Update visualizations immediately
                self.update_gps_map()
                self.update_speed_plot()
            else:
                self.gps_data = pd.DataFrame()
                self.status_label.setText("No valid GPS data found for this event")
                print("No valid GPS messages found")

        self.update_variable_selector()
        self.status_label.setText(f"Event loaded: {event_name}")
        self.update_plot()

    def update_plot(self):
        """Update main data plot with safe parsing"""
        if self.event_data.empty or self.variable_selector.currentIndex() == -1:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No data to display", 
                        ha='center', va='center', fontsize=12, color=GRAPH_TEXT_COLOR)
            self.canvas.draw()
            return

        print(f"Updating plot for {self.type_selector.currentText()} - {self.variable_selector.currentText()}")

        self.ax.clear()
        sensor_type = self.type_selector.currentText()
        variable_name = self.variable_selector.currentText()

        try:
            # Handle $GNRMC using pre-parsed gps_data
            if sensor_type == "$GNRMC":
                if self.gps_data.empty or 'Timestamp' not in self.gps_data.columns:
                    self.ax.text(0.5, 0.5, "No valid GPS data available", 
                            ha='center', va='center', fontsize=12, color=GRAPH_TEXT_COLOR)
                    self.canvas.draw()
                    return

                if variable_name == "Speed_Knots":
                    y_values = self.gps_data['Speed_Knots']
                    timestamps = self.gps_data['Timestamp']
                elif variable_name == "Angle":
                    y_values = self.gps_data['Angle']
                    timestamps = self.gps_data['Timestamp']
                else:
                    raise ValueError(f"Invalid variable for $GNRMC: {variable_name}")
            else:
                # Handle FFS and IMU
                var_index = VARIABLES[sensor_type].index(variable_name)
                print("First rows of 'Data' column:")
                print(self.event_data['Data'].head())

                # Split and convert data
                data_split = self.event_data['Data'].str.split(',', expand=True)
                if data_split.shape[1] <= var_index:
                    raise ValueError(f"Data has only {data_split.shape[1]} columns, but variable index is {var_index}")

                data_split = data_split.apply(pd.to_numeric, errors='coerce')
                y_values = data_split.iloc[:, var_index]
                timestamps = self.event_data['Timestamp']

                # Drop NaN values
                valid_data = pd.DataFrame({'Timestamp': timestamps, 'Value': y_values}).dropna()
                if valid_data.empty:
                    self.ax.text(0.5, 0.5, "No valid data to plot after cleaning", 
                            ha='center', va='center', fontsize=12, color=GRAPH_TEXT_COLOR)
                    self.canvas.draw()
                    return

                timestamps = valid_data['Timestamp']
                y_values = valid_data['Value']

            # Plot the data
            self.ax.plot(
                timestamps,
                y_values,
                label=f"{variable_name}",
                linewidth=2,
                color=GRAPH_LINE_COLOR
            )

            # Add cursor line
            if not timestamps.empty:
                self.cursor_line = self.ax.axvline(
                    x=timestamps.iloc[0],
                    color=GRAPH_HIGHLIGHT_COLOR,
                    linestyle='--',
                    linewidth=1.5
                )

            # Format the plot
            self.ax.legend(loc='upper right')
            self.ax.set_xlabel("Time", color=GRAPH_TEXT_COLOR)
            self.ax.set_ylabel(variable_name, color=GRAPH_TEXT_COLOR)
            self.ax.grid(True, color=GRAPH_GRID_COLOR, linestyle='--', alpha=0.7)
            self.ax.set_title(f"{sensor_type} - {variable_name}", color=GRAPH_TEXT_COLOR, pad=10)
            plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right', color=GRAPH_TEXT_COLOR)
            plt.setp(self.ax.get_yticklabels(), color=GRAPH_TEXT_COLOR)

            self.figure.tight_layout()
            self.canvas.draw()
            print("Plot updated successfully")

        except Exception as e:
            print(f"Error updating plot: {e}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error plotting: {str(e)}", 
                        ha='center', va='center', fontsize=12, color=GRAPH_TEXT_COLOR)
            self.canvas.draw()

    def update_frame(self):
        """Update video frame and synchronized visualizations"""
        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_playback()
            return

        self.current_frame += 1

        # Process and display video frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        
        # Maintain aspect ratio
        scale_factor = min(self.video_frame_label.width() / width, 
                        self.video_frame_label.height() / height)
        frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.video_frame_label.setPixmap(QPixmap.fromImage(image))

        # Get current video time
        current_time_sec = self.video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        start_video_sec = self.parse_time(self.current_event['Start'])
        relative_time = current_time_sec - start_video_sec
        
        print(f"\nFrame {self.current_frame}: Video time: {current_time_sec:.2f}s, Relative time: {relative_time:.2f}s")

        # Calculate reference time for data synchronization
        if not self.event_data.empty and 'Timestamp' in self.event_data.columns:
            event_reference_time = self.event_data['Timestamp'].iloc[0] + pd.to_timedelta(relative_time, unit='s')
            print(f"Event reference time: {event_reference_time}")
            
            # Update plot cursor position
            if self.cursor_line:
                self.cursor_line.set_xdata([event_reference_time])
                self.canvas.draw()
                print("Cursor position updated")

            # Update GPS visualization
            if not self.gps_data.empty and 'Timestamp' in self.gps_data.columns:
                # Find closest GPS data point
                time_deltas = (self.gps_data['Timestamp'] - event_reference_time).abs()
                closest_index = time_deltas.idxmin()
                print(f"Closest GPS point: {closest_index} at {self.gps_data['Timestamp'].iloc[closest_index]}")
                
                # Update map with progress indicator
                self.gps_map_ax.clear()
                self.gps_map_ax.add_feature(cfeature.LAND, color='#f0f0f0')
                self.gps_map_ax.add_feature(cfeature.OCEAN, color='#e6f3ff')
                self.gps_map_ax.add_feature(cfeature.COASTLINE, edgecolor=GRAPH_TEXT_COLOR)
                self.gps_map_ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor=GRAPH_TEXT_COLOR)
                
                # Plot completed path in color
                completed_segments = np.array([self.gps_data['Longitude'].iloc[:closest_index+1], 
                                             self.gps_data['Latitude'].iloc[:closest_index+1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([completed_segments[:-1], completed_segments[1:]], axis=1)
                
                norm = plt.Normalize(self.gps_data['Speed_Knots'].min(), self.gps_data['Speed_Knots'].max())
                cmap = cm.get_cmap('viridis')
                
                lc = LineCollection(segments, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
                lc.set_array(self.gps_data['Speed_Knots'].iloc[:closest_index+1])
                lc.set_linewidth(2)
                self.gps_map_ax.add_collection(lc)
                
                # Plot remaining path in light gray
                if closest_index < len(self.gps_data) - 1:
                    remaining_segments = np.array([self.gps_data['Longitude'].iloc[closest_index:], 
                                                 self.gps_data['Latitude'].iloc[closest_index:]]).T.reshape(-1, 1, 2)
                    remaining_segments = np.concatenate([remaining_segments[:-1], remaining_segments[1:]], axis=1)
                    
                    lc_remaining = LineCollection(remaining_segments, color='#cccccc', transform=ccrs.PlateCarree())
                    lc_remaining.set_linewidth(1)
                    self.gps_map_ax.add_collection(lc_remaining)
                
                # Highlight current position
                self.gps_map_ax.plot(
                    self.gps_data['Longitude'].iloc[closest_index],
                    self.gps_data['Latitude'].iloc[closest_index],
                    'ro',
                    markersize=8,
                    transform=ccrs.PlateCarree(),
                    label='Current Position'
                )
                
                # Add start and end markers
                self.gps_map_ax.plot(
                    self.gps_data['Longitude'].iloc[0],
                    self.gps_data['Latitude'].iloc[0],
                    'go',
                    markersize=8,
                    transform=ccrs.PlateCarree(),
                    label='Start'
                )
                
                self.gps_map_ax.plot(
                    self.gps_data['Longitude'].iloc[-1],
                    self.gps_data['Latitude'].iloc[-1],
                    'yo',
                    markersize=8,
                    transform=ccrs.PlateCarree(),
                    label='End'
                )
                
                self.gps_map_ax.legend(loc='upper right')
                
                # Set map extent with padding
                lat_pad = max(0.01, 0.01 * (self.gps_data['Latitude'].max() - self.gps_data['Latitude'].min()))
                lon_pad = max(0.01, 0.01 * (self.gps_data['Longitude'].max() - self.gps_data['Longitude'].min()))
                
                self.gps_map_ax.set_extent([
                    self.gps_data['Longitude'].min() - lon_pad,
                    self.gps_data['Longitude'].max() + lon_pad,
                    self.gps_data['Latitude'].min() - lat_pad,
                    self.gps_data['Latitude'].max() + lat_pad
                ], crs=ccrs.PlateCarree())
                
                self.gps_map_fig.tight_layout()
                self.gps_map_canvas.draw()
                print("GPS map updated")

            # Update speed plot
            if not self.gps_data.empty:
                self.speed_ax.clear()
                
                # Convert speed based on selected unit
                if self.speed_unit == 'knots':
                    speed = self.gps_data['Speed_Knots']
                    unit_label = "Speed (knots)"
                elif self.speed_unit == 'kmh':
                    speed = self.gps_data['Speed_Knots'] * 1.852
                    unit_label = "Speed (km/h)"
                else:  # m/s
                    speed = self.gps_data['Speed_Knots'] * 0.514444
                    unit_label = "Speed (m/s)"
                
                # Create color segments for completed path
                completed_points = np.array([self.gps_data['Timestamp'].iloc[:closest_index+1], 
                                           speed.iloc[:closest_index+1]]).T.reshape(-1, 1, 2)
                completed_segments = np.concatenate([completed_points[:-1], completed_points[1:]], axis=1)
                
                # Create color segments for remaining path
                if closest_index < len(self.gps_data) - 1:
                    remaining_points = np.array([self.gps_data['Timestamp'].iloc[closest_index:], 
                                               speed.iloc[closest_index:]]).T.reshape(-1, 1, 2)
                    remaining_segments = np.concatenate([remaining_points[:-1], remaining_points[1:]], axis=1)
                
                # Create colormap
                norm = plt.Normalize(speed.min(), speed.max())
                cmap = cm.get_cmap('viridis')
                
                # Plot completed path
                lc_completed = LineCollection(completed_segments, cmap=cmap, norm=norm)
                lc_completed.set_array(speed.iloc[:closest_index+1])
                lc_completed.set_linewidth(2)
                self.speed_ax.add_collection(lc_completed)
                
                # Plot remaining path in light gray
                if closest_index < len(self.gps_data) - 1:
                    lc_remaining = LineCollection(remaining_segments, color='#cccccc')
                    lc_remaining.set_linewidth(1)
                    self.speed_ax.add_collection(lc_remaining)
                
                # Highlight current position
                self.speed_ax.plot(
                    [self.gps_data['Timestamp'].iloc[closest_index]],
                    [speed.iloc[closest_index]],
                    'ro',
                    markersize=8
                )
                
                # Add colorbar
                if hasattr(self, 'colorbar'):
                    self.colorbar.remove()
                self.colorbar = self.speed_fig.colorbar(lc_completed, ax=self.speed_ax, orientation='horizontal', pad=0.2)
                self.colorbar.set_label(unit_label, color=GRAPH_TEXT_COLOR)
                self.colorbar.ax.xaxis.set_tick_params(color=GRAPH_TEXT_COLOR)
                plt.setp(plt.getp(self.colorbar.ax.axes, 'xticklabels'), color=GRAPH_TEXT_COLOR)
                
                # Set axis limits
                self.speed_ax.set_xlim(self.gps_data['Timestamp'].min(), self.gps_data['Timestamp'].max())
                self.speed_ax.set_ylim(speed.min() * 0.95, speed.max() * 1.05)
                
                # Formatting
                self.speed_ax.set_xlabel("Time", color=GRAPH_TEXT_COLOR)
                self.speed_ax.set_ylabel(unit_label, color=GRAPH_TEXT_COLOR)
                self.speed_ax.grid(True, color=GRAPH_GRID_COLOR, linestyle='--', alpha=0.7)
                self.speed_ax.set_title(f"GPS Speed ({unit_label})", color=GRAPH_TEXT_COLOR, pad=10)
                
                # Format ticks
                plt.setp(self.speed_ax.get_xticklabels(), rotation=45, ha='right', color=GRAPH_TEXT_COLOR)
                plt.setp(self.speed_ax.get_yticklabels(), color=GRAPH_TEXT_COLOR)
                
                self.speed_fig.tight_layout()
                self.speed_canvas.draw()
                print("Speed plot updated")

            # Update surfboard visualization
            if not self.imu_data.empty and 'Timestamp' in self.imu_data.columns:
                time_deltas = (self.imu_data['Timestamp'] - event_reference_time).abs()
                closest_index = time_deltas.idxmin()
                
                yaw = self.imu_data.loc[closest_index, 'Yaw']
                pitch = self.imu_data.loc[closest_index, 'Pitch']
                roll = self.imu_data.loc[closest_index, 'Roll']
                
                self.surfboard_viz.update_orientation(yaw, pitch, roll)
                print(f"Surfboard updated - Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}")

        # Check for end of playback
        stop_video_seconds = self.parse_time(self.current_event['Stop'])
        if current_time_sec >= stop_video_seconds:
            print("Reached end of playback")
            self.stop_playback()

    def parse_time(self, time_str):
        """Convert time string in HH:MM:SS format to seconds"""
        try:
            if pd.isna(time_str):  # Check for NaN values
                return 0
            parts = list(map(int, str(time_str).split(':')))  # Split into hours, minutes, seconds
            return parts[0] * 3600 + parts[1] * 60 + parts[2]  # Convert to seconds
        except Exception as e:
            print(f"Error parsing time: {e}")
            return 0


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set modern style
    app.setStyle('Fusion')
    
    # Set custom palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # Set font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    player = VideoDataPlayer()
    player.show()
    sys.exit(app.exec_())