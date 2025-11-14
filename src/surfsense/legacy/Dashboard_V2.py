import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QFileDialog, QSplitter, 
                             QGroupBox, QFormLayout, QSlider, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLLinePlotItem, GLMeshItem
from OpenGL.GL import *
from OpenGL.GLU import *

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

        # Desenho manual da prancha conforme imagem enviada
        verts = np.array([
            [0.0, 0.0, 0.0],      # Center point (opcional)
            [-0.8, -0.07, -0.05],   # Tail left (negativo para concavidade para baixo)
            [-0.8, 0.07, -0.05],    # Tail right
            [-0.5, -0.3, -0.1],    # Low mid left
            [-0.5, 0.3, -0.1],     # Low mid right
            [0.3, -0.3, -0.1],     # High mid left
            [0.3, 0.3, -0.1],      # High mid right
            [1.0, 0.0, 0.0],       # Nose point
        ])

        # Centraliza no centro geométrico
        center = verts.mean(axis=0)
        verts -= center

        # Define as faces conectando os vértices
        faces = np.array([
            [1, 2, 3], [2, 4, 3],    # Tail para Low Mid
            [3, 4, 5], [4, 6, 5],    # Low Mid para High Mid
            [5, 6, 7],              # High Mid para Nose
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

        # Eixos coordenados
        axis_x = GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(1, 0, 0, 1), width=2)
        axis_y = GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 1, 0]]), color=(0, 1, 0, 1), width=2)
        axis_z = GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 1]]), color=(0, 0, 1, 1), width=2)

        self.addItem(axis_x)
        self.addItem(axis_y)
        self.addItem(axis_z)

        # Câmera inicial
        self.setCameraPosition(distance=3, elevation=20, azimuth=45)

    def update_orientation(self, yaw, pitch, roll):
        """Atualiza orientação da prancha baseada no IMU"""
        self.surfboard.resetTransform()

        self.surfboard.rotate(roll, 1, 0, 0, local=True)    # Roll
        self.surfboard.rotate(pitch, 0, 1, 0, local=True)   # Pitch
        self.surfboard.rotate(yaw, 0, 0, 1, local=True)     # Yaw


class VideoDataPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surf Data Visualization Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize variables
        self.conn = None
        self.trimming_df = None
        self.video_capture = None
        self.video_fps = 30
        self.playback_speed = 1.0
        self.is_playing = False
        self.timer = QTimer()
        self.event_data = pd.DataFrame()
        self.imu_data = pd.DataFrame()
        self.cursor_line = None
        self.current_event = None
        
        self.init_ui()
        self.timer.timeout.connect(self.update_frame)

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
        main_splitter.setSizes([1000, 600])
        
        # ===== DATA PANEL =====
        # Control panel group
        control_group = QGroupBox("Data Controls")
        control_layout = QVBoxLayout()
        
        # File import buttons
        file_buttons = QHBoxLayout()
        self.db_button = QPushButton("Load Database")
        self.db_button.clicked.connect(self.load_database)
        file_buttons.addWidget(self.db_button)
        
        self.trim_button = QPushButton("Load Events CSV")
        self.trim_button.clicked.connect(self.load_trimming_csv)
        file_buttons.addWidget(self.trim_button)
        
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
        event_form.addRow("Sensor Type:", self.type_selector)
        self.type_selector.currentTextChanged.connect(self.update_variable_selector)
        
        # Variable selection
        self.variable_selector = QComboBox()
        event_form.addRow("Variable:", self.variable_selector)
        self.variable_selector.currentTextChanged.connect(self.update_plot)
        
        control_layout.addLayout(event_form)
        control_group.setLayout(control_layout)
        data_layout.addWidget(control_group)
        
        # Plot area
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.figure.set_facecolor('#f0f0f0')
        self.ax.set_facecolor('#fafafa')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        data_layout.addWidget(self.toolbar)
        data_layout.addWidget(self.canvas)
        
        # ===== RIGHT PANEL =====
        # Video control group
        video_control_group = QGroupBox("Video Controls")
        video_control_layout = QVBoxLayout()
        
        # Video import button
        self.video_button = QPushButton("Load Video File")
        self.video_button.clicked.connect(self.load_video)
        video_control_layout.addWidget(self.video_button)
        
        # Playback controls
        playback_buttons = QHBoxLayout()
        self.play_pause_button = QPushButton("▶ Play")
        self.play_pause_button.clicked.connect(self.toggle_playback)
        playback_buttons.addWidget(self.play_pause_button)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        
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
        
        # Initialize variable selector
        self.update_variable_selector()

    def update_playback_speed(self):
        """Update playback speed based on slider value"""
        self.playback_speed = self.speed_slider.value() / 10.0
        self.speed_label.setText(f"{self.playback_speed:.1f}x")
        
        if self.is_playing:
            self.timer.stop()
            self.timer.start(int(1000 / (self.video_fps * self.playback_speed)))

    def toggle_playback(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def load_database(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SQLite Database", 
            os.getcwd(), "Database Files (*.db)"
        )
        if file_path:
            self.conn = sqlite3.connect(file_path)
            self.status_label.setText(f"Database loaded: {os.path.basename(file_path)}")
            print("Database loaded successfully.")

    def load_trimming_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Events CSV File", 
            os.getcwd(), "CSV Files (*.csv)"
        )
        if file_path:
            self.trimming_df = pd.read_csv(file_path)
            self.event_selector.clear()
            self.event_selector.addItems(self.trimming_df['Events'].dropna().unique())
            self.status_label.setText(f"Events loaded: {os.path.basename(file_path)}")
            print("Events CSV loaded successfully.")

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", 
            os.getcwd(), "Video Files (*.mp4 *.avi)"
        )
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.status_label.setText(f"Video loaded: {os.path.basename(file_path)} @ {self.video_fps:.1f} FPS")
            print(f"Video loaded at {self.video_fps} FPS.")

    def update_variable_selector(self):
        sensor_type = self.type_selector.currentText()
        self.variable_selector.clear()
        self.variable_selector.addItems(VARIABLES[sensor_type])
        self.update_plot()

    def prepare_event(self):
        if self.conn is None or self.trimming_df is None:
            self.status_label.setText("Please load database and events CSV first")
            return

        event_name = self.event_selector.currentText()
        if not event_name:
            return

        self.current_event = self.trimming_df[self.trimming_df['Events'] == event_name].iloc[0]

        # Load all sensor data for this event
        df = pd.read_sql_query(
            "SELECT * FROM Sensor_Data WHERE Event_ID = (SELECT Event_ID FROM Events WHERE Event_Name = ?)",
            self.conn, params=(event_name,)
        )

        if df.empty:
            self.status_label.setText(f"No data found for event: {event_name}")
            print("No data for this event.")
            self.event_data = pd.DataFrame()
            self.imu_data = pd.DataFrame()
            return

        # Process timestamps
        df['Millisec'] = df['Millisec'].fillna(0).astype(float).astype(int).astype(str).str.zfill(3)
        df['Timestamp'] = pd.to_datetime(
            df['Time'] + '.' + df['Millisec'], 
            format='%H:%M:%S.%f', 
            errors='coerce'
        )
        df = df.dropna(subset=['Timestamp'])

        # Store IMU data separately for surfboard visualization
        self.imu_data = df[df['Data_Type'] == "IMU"].copy()
        if not self.imu_data.empty:
            # Parse IMU data columns
            imu_values = self.imu_data['Data'].str.split(',', expand=True).astype(float)
            self.imu_data['Yaw'] = imu_values.iloc[:, VARIABLES["IMU"].index("Yaw")]
            self.imu_data['Pitch'] = imu_values.iloc[:, VARIABLES["IMU"].index("Pitch")]
            self.imu_data['Roll'] = imu_values.iloc[:, VARIABLES["IMU"].index("Roll")]

        self.event_data = df[df['Data_Type'] == self.type_selector.currentText()]
        self.status_label.setText(f"Event loaded: {event_name}")
        print("Event prepared successfully.")
        self.update_plot()

    def update_plot(self):
        if self.event_data.empty or self.variable_selector.currentIndex() == -1:
            return

        self.ax.clear()
        sensor_type = self.type_selector.currentText()
        var_index = VARIABLES[sensor_type].index(self.variable_selector.currentText())

        data_series = self.event_data['Data'].str.split(',', expand=True).astype(float).iloc[:, var_index]
        
        # Enhanced plot styling
        self.ax.plot(
            self.event_data['Timestamp'], 
            data_series, 
            label=f"{self.variable_selector.currentText()}",
            linewidth=2,
            color='#1f77b4'
        )
        
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel("Time", fontsize=10)
        self.ax.set_ylabel(self.variable_selector.currentText(), fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_title(f"{sensor_type} - {self.variable_selector.currentText()}", fontsize=12)
        
        # Format x-axis ticks for better readability
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')

        # Add cursor line
        if not self.event_data.empty:
            self.cursor_line = self.ax.axvline(
                x=self.event_data['Timestamp'].iloc[0], 
                color='red', 
                linestyle='--',
                linewidth=1.5
            )

        self.figure.tight_layout()
        self.canvas.draw()

    def start_playback(self):
        if self.video_capture and self.current_event is not None:
            start_video_seconds = self.parse_time(self.current_event['Start'])
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, start_video_seconds * 1000)
            self.timer.start(int(1000 / (self.video_fps * self.playback_speed)))
            self.is_playing = True
            self.play_pause_button.setText("⏸ Pause")
            self.status_label.setText("Playing...")

    def pause_playback(self):
        self.timer.stop()
        self.is_playing = False
        self.play_pause_button.setText("▶ Play")
        self.status_label.setText("Paused")

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.is_playing = False
            self.play_pause_button.setText("▶ Play")
            self.status_label.setText("Playback finished")
            return

        # Process and display video frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        
        # Maintain aspect ratio while fitting to available space
        label_width = self.video_frame_label.width()
        label_height = self.video_frame_label.height()
        
        scale_factor = min(label_width / width, label_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        frame = cv2.resize(frame, (new_width, new_height))
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.video_frame_label.setPixmap(QPixmap.fromImage(image))

        # Get current video time
        current_time_sec = self.video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        start_video_sec = self.parse_time(self.current_event['Start'])
        relative_time = current_time_sec - start_video_sec
        
        # Update plot cursor position
        if not self.event_data.empty:
            event_time = self.event_data['Timestamp'].iloc[0] + pd.to_timedelta(relative_time, unit='s')
            if self.cursor_line:
                self.cursor_line.set_xdata([event_time])
            self.canvas.draw()

        # Update surfboard visualization if IMU data is available
        if not self.imu_data.empty:
            # Find the closest timestamp in IMU data
            time_deltas = (self.imu_data['Timestamp'] - (self.event_data['Timestamp'].iloc[0] + 
                         pd.to_timedelta(relative_time, unit='s'))).abs()
            closest_index = time_deltas.idxmin()
            
            yaw = self.imu_data.loc[closest_index, 'Yaw']
            pitch = self.imu_data.loc[closest_index, 'Pitch']
            roll = self.imu_data.loc[closest_index, 'Roll']
            
            self.surfboard_viz.update_orientation(yaw, pitch, roll)

        # Check for end of playback range
        stop_video_seconds = self.parse_time(self.current_event['Stop'])
        if current_time_sec >= stop_video_seconds:
            self.timer.stop()
            self.is_playing = False
            self.play_pause_button.setText("▶ Play")
            self.status_label.setText("Playback completed")

    def parse_time(self, time_str):
        try:
            if pd.isna(time_str):
                return 0
            parts = list(map(int, str(time_str).split(':')))
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        except:
            return 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set a modern style
    app.setStyle('Fusion')
    
    player = VideoDataPlayer()
    player.show()
    sys.exit(app.exec_())