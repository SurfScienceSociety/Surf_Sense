import sys
import os
import csv
import re
import sqlite3
import pandas as pd
import numpy as np
import cv2
import shutil
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QComboBox, QPushButton, QFileDialog, QProgressBar, QGroupBox,
                             QTabWidget, QScrollArea, QSizePolicy, QSpacerItem, QStyle)  # Added QStyle here
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import seaborn as sns

# Sensor variables
SENSOR_VARIABLES = {
    "FFS": ["X_adc", "Y_adc", "X_load", "Y_load", "Res_Force", "X_Flow", "Y_Flow", "Res_Flow", "Res_Angle", "Temperature"],
    "IMU": ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mag_X", "Mag_Y", "Mag_Z", "Yaw", "Pitch", "Roll"],
    "$GNRMC": ["Speed_Knots", "Angle"]
}

class ModernFigureCanvas(FigureCanvas):
    """Custom FigureCanvas with modern styling"""
    def __init__(self, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setup_style()
        
    def setup_style(self):
        """Apply modern styling to the figure"""
        self.fig.patch.set_facecolor('#F5F7FA')
        self.fig.patch.set_alpha(0.0)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#FFFFFF')
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.ax.tick_params(colors='#4A4A4A')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('#D3D3D3')
        self.ax.spines['left'].set_color('#D3D3D3')

class SurfAnalysisApp(QMainWindow):
    update_progress = pyqtSignal(int, str)
    data_loaded = pyqtSignal()

    def update_variable_combo(self):
        """Update the variable combo box based on selected sensor type"""
        self.var_combo.clear()
        sensor_type = self.sensor_combo.currentText()
        
        if sensor_type in SENSOR_VARIABLES:
            self.var_combo.addItems(SENSOR_VARIABLES[sensor_type])
        
        # Enable/disable the variable combo based on whether items exist
        self.var_combo.setEnabled(self.var_combo.count() > 0)
        
        # Update the plot if we already have data loaded
        if hasattr(self, 'event_data') and not self.event_data.empty:
            self.update_plot()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SurfSense Pro - Advanced Surfing Analytics")
        self.resize(1600, 1000)
        
        # Set window icon
        self.setWindowIcon(QIcon(self.style().standardPixmap(QStyle.SP_DesktopIcon)))
        
        # Application state
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.abspath(os.path.join(self.base_dir, 'data'))
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_conn = None
        self.video_capture = None
        self.event_data = pd.DataFrame()
        self.current_event = None
        self.current_frame = None
        self.playback_speed = 1.0
        self.cursor_line = None
        self.video_markers = []

        # Initialize UI
        self.init_ui()
        self.init_connections()
        self.apply_styles()

    def init_ui(self):
        """Initialize the modern user interface"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(15)

        # Left Panel: Navigation and Processing
        self.nav_panel = QGroupBox("Processing Pipeline")
        self.nav_panel.setFixedWidth(300)
        self.nav_layout = QVBoxLayout()
        self.nav_layout.setSpacing(15)
        
        # Add application logo/header
        self.app_header = QLabel("SurfSense Pro")
        self.app_header.setAlignment(Qt.AlignCenter)
        self.app_header.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        self.nav_layout.addWidget(self.app_header)
        
        # Add processing steps
        self.btn_phase0 = self.create_step_button("1. Preprocess Raw Data", "Import and clean raw sensor data")
        self.btn_phase1 = self.create_step_button("2. Create Database", "Build database from processed data")
        self.btn_phase2 = self.create_step_button("3. Clean Data", "Remove outliers and interpolate")
        self.btn_phase3 = self.create_step_button("4. Visualize Data", "Interactive data exploration")
        
        self.nav_layout.addWidget(self.btn_phase0)
        self.nav_layout.addWidget(self.btn_phase1)
        self.nav_layout.addWidget(self.btn_phase2)
        self.nav_layout.addWidget(self.btn_phase3)
        
        # Add status area
        self.status_group = QGroupBox("Processing Status")
        self.status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        
        self.status_layout.addWidget(self.progress_bar)
        self.status_layout.addWidget(self.status_label)
        self.status_group.setLayout(self.status_layout)
        self.nav_layout.addWidget(self.status_group)
        
        self.nav_layout.addStretch()
        self.nav_panel.setLayout(self.nav_layout)
        self.main_layout.addWidget(self.nav_panel)

        # Right Panel: Visualization and Analysis
        self.viz_panel = QTabWidget()
        self.viz_panel.setDocumentMode(True)
        self.viz_panel.setTabPosition(QTabWidget.North)
        
        # Tab 1: Data Visualization
        self.tab_viz = QWidget()
        self.tab_viz_layout = QVBoxLayout()
        self.tab_viz_layout.setContentsMargins(5, 5, 5, 5)
        
        # Visualization controls
        self.viz_controls = QGroupBox("Visualization Controls")
        self.viz_control_layout = QGridLayout()
        
        # Row 1: Event selection
        self.event_label = QLabel("Event:")
        self.event_combo = QComboBox()
        self.event_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Row 2: Sensor and variable selection
        self.sensor_label = QLabel("Sensor:")
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(SENSOR_VARIABLES.keys())
        
        self.var_label = QLabel("Variable:")
        self.var_combo = QComboBox()
        
        # Row 3: Plot controls
        self.btn_export_plot = QPushButton("Export Plot")
        self.btn_export_plot.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        
        self.btn_show_stats = QPushButton("Show Stats")
        self.btn_show_stats.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        
        # Add widgets to grid
        self.viz_control_layout.addWidget(self.event_label, 0, 0)
        self.viz_control_layout.addWidget(self.event_combo, 0, 1, 1, 3)
        
        self.viz_control_layout.addWidget(self.sensor_label, 1, 0)
        self.viz_control_layout.addWidget(self.sensor_combo, 1, 1)
        self.viz_control_layout.addWidget(self.var_label, 1, 2)
        self.viz_control_layout.addWidget(self.var_combo, 1, 3)
        
        self.viz_control_layout.addWidget(self.btn_export_plot, 2, 0, 1, 2)
        self.viz_control_layout.addWidget(self.btn_show_stats, 2, 2, 1, 2)
        
        self.viz_controls.setLayout(self.viz_control_layout)
        self.tab_viz_layout.addWidget(self.viz_controls)
        
        # Main plot area
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout()
        
        # Create figure with modern theme
        plt.style.use('ggplot')  # Changed from 'seaborn-darkgrid'
        sns.set_palette("deep")

        self.figure = plt.figure(facecolor='#F5F7FA')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)
        self.plot_container.setLayout(self.plot_layout)
        
        self.tab_viz_layout.addWidget(self.plot_container)
        self.tab_viz.setLayout(self.tab_viz_layout)
        self.viz_panel.addTab(self.tab_viz, "Data Visualization")
        
        # Tab 2: Video Synchronization
        self.tab_video = QWidget()
        self.tab_video_layout = QVBoxLayout()
        
        # Video controls
        self.video_controls = QGroupBox("Video Controls")
        self.video_control_layout = QGridLayout()
        
        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_video.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        
        self.btn_play = QPushButton("Play")
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(1)
        
        self.video_position = QProgressBar()
        self.video_position.setTextVisible(False)
        
        self.video_time = QLabel("00:00:00 / 00:00:00")
        self.video_time.setAlignment(Qt.AlignCenter)
        
        self.video_control_layout.addWidget(self.btn_load_video, 0, 0, 1, 2)
        self.video_control_layout.addWidget(self.btn_play, 1, 0)
        self.video_control_layout.addWidget(self.btn_pause, 1, 1)
        self.video_control_layout.addWidget(QLabel("Speed:"), 2, 0)
        self.video_control_layout.addWidget(self.speed_combo, 2, 1)
        self.video_control_layout.addWidget(self.video_position, 3, 0, 1, 2)
        self.video_control_layout.addWidget(self.video_time, 4, 0, 1, 2)
        
        self.video_controls.setLayout(self.video_control_layout)
        self.tab_video_layout.addWidget(self.video_controls)
        
        # Video display area
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setStyleSheet("background-color: black;")
        
        self.tab_video_layout.addWidget(self.video_display, 1)
        self.tab_video.setLayout(self.tab_video_layout)
        self.viz_panel.addTab(self.tab_video, "Video Sync")
        
        # Tab 3: Event Statistics
        self.tab_stats = QWidget()
        self.tab_stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("Event Statistics will appear here")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.tab_stats_layout.addWidget(self.stats_label)
        
        self.tab_stats.setLayout(self.tab_stats_layout)
        self.viz_panel.addTab(self.tab_stats, "Statistics")
        
        self.main_layout.addWidget(self.viz_panel, 1)
        
        # Initialize playback timer
        self.playback_timer = QTimer()
        self.playback_timer.setInterval(33)  # ~30fps
    
    def run_phase0(self):
        """Preprocess raw data"""
        try:
            self.update_progress.emit(0, "Selecting raw data file...")
            raw_file = self.select_file("Select Raw Data File", "", "Text Files (*.csv *.txt);;All Files (*)")
            if not raw_file:
                self.update_progress.emit(0, "No file selected.")
                return

            dest_file = os.path.join(self.data_dir, os.path.basename(raw_file))
            shutil.copy(raw_file, dest_file)
            clean_file = self.force_data_folder(dest_file, 'clean')
            shutil.copy(raw_file, clean_file)

            self.update_progress.emit(20, "Filtering data...")
            filtered_lines = self.filter_data(clean_file)
            with open(clean_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(filtered_lines)

            self.update_progress.emit(40, "Adding timestamps...")
            timestamp_file = self.add_timestamps(clean_file)

            self.update_progress.emit(60, "Reordering timestamps...")
            reordered_file = self.reorder_timestamps(timestamp_file)

            self.update_progress.emit(80, "Trimming to valid GPS...")
            self.final_file = self.trim_to_valid_gps(reordered_file)

            self.update_progress.emit(100, "Phase 0 complete - Data preprocessed successfully!")
            self.btn_phase1.setEnabled(True)
        except Exception as e:
            self.update_progress.emit(0, f"Error in Phase 0: {str(e)}")

    def run_phase1(self):
        """Create database from processed data"""
        if not hasattr(self, 'final_file'):
            self.update_progress.emit(0, "Error: Run Phase 0 first.")
            return

        try:
            self.update_progress.emit(0, "Selecting trimming file...")
            trim_file = self.select_file("Select Trimming CSV", self.data_dir, "CSV Files (*.csv)")
            if not trim_file:
                self.update_progress.emit(0, "No trimming file selected.")
                return

            self.update_progress.emit(20, "Selecting database location...")
            db_path = self.select_save_file("Save Database As", 
                                        os.path.join(self.data_dir, "surf_data.db"), 
                                        "Database Files (*.db)")
            if not db_path:
                self.update_progress.emit(0, "No database location selected.")
                return

            self.update_progress.emit(40, "Processing data...")
            self.process_database(self.final_file, trim_file, db_path)
            self.db_path = db_path
            self.db_conn = sqlite3.connect(db_path)
            self.update_progress.emit(100, "Phase 1 complete: Database created.")
            self.btn_phase2.setEnabled(True)
        except Exception as e:
            self.update_progress.emit(0, f"Error in Phase 1: {str(e)}")

    def run_phase2(self):
        """Clean the database"""
        if not self.db_conn:
            self.update_progress.emit(0, "Error: Run Phase 1 first.")
            return

        try:
            self.update_progress.emit(0, "Starting data cleaning...")
            self.clean_data()
            self.update_progress.emit(100, "Phase 2 complete: Data cleaned.")
            self.btn_phase3.setEnabled(True)
        except Exception as e:
            self.update_progress.emit(0, f"Error in Phase 2: {str(e)}")

    def run_phase3(self):
        """Load visualization"""
        if not self.db_conn:
            self.update_progress.emit(0, "Error: Run Phase 2 first.")
            return
            
        self.update_progress.emit(100, "Phase 3 ready: Visualization loaded.")
        self.load_events()

    def create_step_button(self, text, tooltip):
        """Create a styled step button"""
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                text-align: left;
                border-left: 4px solid #3498DB;
            }
            QPushButton:hover {
                background-color: #E1F0FA;
            }
        """)
        return btn

    def apply_styles(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F7FA;
            }
            QGroupBox {
                font: bold 12px 'Segoe UI';
                border: 1px solid #D6D6D6;
                border-radius: 6px;
                margin-top: 15px;
                padding: 10px 5px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #2C3E50;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font: 11px 'Segoe UI';
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #D3D3D3;
                border-radius: 4px;
                background-color: #FFFFFF;
                min-width: 100px;
            }
            QComboBox:hover {
                border: 1px solid #3498DB;
            }
            QLabel {
                color: #2C3E50;
                font: 11px 'Segoe UI';
            }
            QProgressBar {
                border: 1px solid #D3D3D3;
                border-radius: 4px;
                text-align: center;
                background-color: #FFFFFF;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #D6D6D6;
                border-radius: 0px;
                background: #FFFFFF;
            }
            QTabBar::tab {
                background: #ECF0F1;
                border: 1px solid #D6D6D6;
                border-bottom: none;
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #2C3E50;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                border-bottom: 2px solid #3498DB;
                color: #3498DB;
            }
            QTabBar::tab:hover {
                background: #E1F0FA;
            }
        """)
        
        # Set font for the application
        font = QFont("Segoe UI", 9)
        self.setFont(font)

    def init_connections(self):
        """Connect signals and slots"""
        # Processing pipeline
        self.btn_phase0.clicked.connect(self.run_phase0)
        self.btn_phase1.clicked.connect(self.run_phase1)  # Make sure run_phase1 exists
        self.btn_phase2.clicked.connect(self.run_phase2)  # Make sure run_phase2 exists
        self.btn_phase3.clicked.connect(self.run_phase3)  # Make sure run_phase3 exists
        
        # Visualization controls
        self.sensor_combo.currentTextChanged.connect(self.update_variable_combo)
        self.event_combo.currentTextChanged.connect(self.load_event_data)
        self.var_combo.currentTextChanged.connect(self.update_plot)
        self.btn_export_plot.clicked.connect(self.export_plot)
        self.btn_show_stats.clicked.connect(self.show_stats)
        
        # Video controls
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_play.clicked.connect(self.start_playback)
        self.btn_pause.clicked.connect(self.pause_playback)
        self.speed_combo.currentTextChanged.connect(self.update_playback_speed)
        self.playback_timer.timeout.connect(self.update_video_frame)
        
        # Progress updates
        self.update_progress.connect(self.update_progress_display)
        self.data_loaded.connect(self.enable_visualization)

    # === Phase 0: Preprocessing ===
    def run_phase0(self):
        """Preprocess raw data"""
        try:
            self.update_progress.emit(0, "Selecting raw data file...")
            raw_file = self.select_file("Select Raw Data File", "", "Text Files (*.csv *.txt);;All Files (*)")
            if not raw_file:
                self.update_progress.emit(0, "No file selected.")
                return

            dest_file = os.path.join(self.data_dir, os.path.basename(raw_file))
            shutil.copy(raw_file, dest_file)
            clean_file = self.force_data_folder(dest_file, 'clean')
            shutil.copy(raw_file, clean_file)

            self.update_progress.emit(20, "Filtering data...")
            filtered_lines = self.filter_data(clean_file)
            with open(clean_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(filtered_lines)

            self.update_progress.emit(40, "Adding timestamps...")
            timestamp_file = self.add_timestamps(clean_file)

            self.update_progress.emit(60, "Reordering timestamps...")
            reordered_file = self.reorder_timestamps(timestamp_file)

            self.update_progress.emit(80, "Trimming to valid GPS...")
            self.final_file = self.trim_to_valid_gps(reordered_file)

            self.update_progress.emit(100, "Phase 0 complete - Data preprocessed successfully!")
            self.btn_phase1.setEnabled(True)
        except Exception as e:
            self.update_progress.emit(0, f"Error in Phase 0: {str(e)}")

    # [Previous processing methods remain the same...]

    # === Visualization Methods ===
    def load_events(self):
        """Load events into combo box with icons"""
        if not self.db_conn:
            return
            
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT Event_Name FROM Events")
        events = cursor.fetchall()
        
        self.event_combo.clear()
        for (event_name,) in events:
            event_type = "Wave" if "Wave" in event_name else "Paddling"
            icon = self.style().standardIcon(QStyle.SP_MediaPlay) if event_type == "Wave" else self.style().standardIcon(QStyle.SP_MediaSeekForward)
            self.event_combo.addItem(icon, f"{event_type}: {event_name}")

    def update_plot(self):
        """Update plot with selected data using modern styling"""
        if self.event_data.empty or self.var_combo.currentIndex() == -1:
            return

        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Apply styling
        ax.set_facecolor('#FFFFFF')
        for spine in ax.spines.values():
            spine.set_edgecolor('#D3D3D3')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        sensor_type = self.sensor_combo.currentText()
        variable = self.var_combo.currentText()
        var_index = SENSOR_VARIABLES[sensor_type].index(variable)

        plot_data = self.event_data.copy()
        if sensor_type == '$GNRMC':
            plot_data['Value'] = plot_data['Data'].apply(lambda x: self.parse_gnrmc(x)[variable] if self.parse_gnrmc(x) else None)
        else:
            data_cols = plot_data['Data'].str.split(',', expand=True).apply(pd.to_numeric, errors='coerce')
            plot_data['Value'] = data_cols[var_index]

        plot_data = plot_data.dropna(subset=['Value'])
        if plot_data.empty:
            ax.text(0.5, 0.5, "No valid data to display", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='#7F8C8D')
            self.canvas.draw()
            return

        # Create plot with modern styling
        line_color = '#3498DB'
        ax.plot(plot_data['Timestamp'], plot_data['Value'], 
               color=line_color, linewidth=2, alpha=0.8)
        
        # Add cursor line for video sync
        if self.cursor_line and self.cursor_line in ax.lines:
            ax.lines.remove(self.cursor_line)
        self.cursor_line = Line2D([], [], color='#E74C3C', linewidth=1.5, linestyle='--')
        ax.add_line(self.cursor_line)
        
        # Format plot
        ax.set_title(f"{self.current_event['name']} - {variable}", 
                    fontsize=14, pad=10, color='#2C3E50')
        ax.set_xlabel("Time", fontsize=12, color='#2C3E50')
        ax.set_ylabel(variable, fontsize=12, color='#2C3E50')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add hover effect
        def on_mouse_move(event):
            if event.inaxes == ax:
                if self.cursor_line:
                    self.cursor_line.set_xdata([event.xdata, event.xdata])
                    self.canvas.draw_idle()
        
        self.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def export_plot(self):
        """Export current plot to file"""
        if not hasattr(self, 'figure') or len(self.figure.axes) == 0:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot As", 
            os.path.join(self.data_dir, f"{self.current_event['name']}_{self.var_combo.currentText()}.png"),
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf)"
        )
        
        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=self.figure.get_facecolor())

    def show_stats(self):
        """Show statistics for current event and variable"""
        if self.event_data.empty or self.var_combo.currentIndex() == -1:
            return
            
        sensor_type = self.sensor_combo.currentText()
        variable = self.var_combo.currentText()
        var_index = SENSOR_VARIABLES[sensor_type].index(variable)

        plot_data = self.event_data.copy()
        if sensor_type == '$GNRMC':
            plot_data['Value'] = plot_data['Data'].apply(lambda x: self.parse_gnrmc(x)[variable] if self.parse_gnrmc(x) else None)
        else:
            data_cols = plot_data['Data'].str.split(',', expand=True).apply(pd.to_numeric, errors='coerce')
            plot_data['Value'] = data_cols[var_index]

        plot_data = plot_data.dropna(subset=['Value'])
        if plot_data.empty:
            return
            
        stats = plot_data['Value'].describe()
        stats_text = (
            f"<h3>Statistics for {variable}</h3>"
            f"<p><b>Event:</b> {self.current_event['name']}</p>"
            f"<p><b>Count:</b> {stats['count']:.0f}</p>"
            f"<p><b>Mean:</b> {stats['mean']:.2f}</p>"
            f"<p><b>Std Dev:</b> {stats['std']:.2f}</p>"
            f"<p><b>Min:</b> {stats['min']:.2f}</p>"
            f"<p><b>25%:</b> {stats['25%']:.2f}</p>"
            f"<p><b>50%:</b> {stats['50%']:.2f}</p>"
            f"<p><b>75%:</b> {stats['75%']:.2f}</p>"
            f"<p><b>Max:</b> {stats['max']:.2f}</p>"
        )
        
        self.stats_label.setText(stats_text)
        self.viz_panel.setCurrentIndex(2)  # Switch to stats tab

    # === Video Methods ===
    def load_video(self):
        """Load video file with preview"""
        video_path = self.select_file(
            "Select Video File", 
            "", 
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if video_path:
            if self.video_capture:
                self.video_capture.release()
                
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                self.update_progress.emit(0, "Error: Could not open video file")
                return
                
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.video_length = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / self.video_fps
            self.playback_timer.setInterval(int(1000 / self.video_fps))
            
            # Update video info
            self.update_video_info()
            
            # Get first frame
            ret, frame = self.video_capture.read()
            if ret:
                self.display_video_frame(frame)
                self.current_frame = frame
                
            # Enable video controls
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(True)
            self.speed_combo.setEnabled(True)

    def update_video_info(self):
        """Update video duration information"""
        if not self.video_capture:
            return
            
        total_seconds = int(self.video_length)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        self.video_time.setText(f"00:00:00 / {total_time}")

    def start_playback(self):
        """Start video playback with synchronization"""
        if not self.video_capture or not self.current_event:
            return
            
        # Calculate start time based on event
        start_time = self.time_to_seconds(self.current_event['start'])
        self.video_capture.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        # Start timer
        self.playback_timer.start()
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def pause_playback(self):
        """Pause video playback"""
        self.playback_timer.stop()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def update_playback_speed(self):
        """Update playback speed based on selection"""
        speed_text = self.speed_combo.currentText()
        self.playback_speed = float(speed_text[:-1])
        interval = int(1000 / (self.video_fps * self.playback_speed))
        self.playback_timer.setInterval(interval)

    def update_video_frame(self):
        """Update video frame with synchronization to plot"""
        if not self.video_capture:
            self.playback_timer.stop()
            return
            
        ret, frame = self.video_capture.read()
        if not ret:
            self.playback_timer.stop()
            return
            
        self.current_frame = frame
        self.display_video_frame(frame)
        
        # Update position indicator
        current_pos = self.video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.video_position.setValue(int((current_pos / self.video_length) * 100))
        
        # Update time display
        current_seconds = int(current_pos)
        hours, remainder = divmod(current_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        current_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        total_seconds = int(self.video_length)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        self.video_time.setText(f"{current_time} / {total_time}")
        
        # Update cursor position on plot if we have event data
        if hasattr(self, 'current_event') and self.current_event:
            event_start = self.time_to_seconds(self.current_event['start'])
            event_time = self.event_data['Timestamp'].iloc[0] + pd.Timedelta(seconds=current_pos - event_start)
            
            if hasattr(self, 'cursor_line') and self.cursor_line:
                self.cursor_line.set_xdata([event_time])
                self.canvas.draw()

    def display_video_frame(self, frame):
        """Display video frame with proper aspect ratio"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Calculate aspect ratio
        target_w = self.video_display.width()
        target_h = self.video_display.height()
        aspect = w / h
        
        if w > target_w or h > target_h:
            if (target_w / aspect) <= target_h:
                w, h = target_w, int(target_w / aspect)
            else:
                h, w = target_h, int(target_h * aspect)
                
        frame = cv2.resize(frame, (w, h))
        q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(q_img))

    # [Previous helper methods remain the same...]

    def enable_visualization(self):
        """Enable visualization controls when data is loaded"""
        self.btn_export_plot.setEnabled(True)
        self.btn_show_stats.setEnabled(True)
        self.btn_load_video.setEnabled(True)

    def update_progress_display(self, value, message):
        """Update progress bar and status with styling"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
        # Update button states based on progress
        self.btn_phase1.setEnabled(hasattr(self, 'final_file'))
        self.btn_phase2.setEnabled(self.db_conn is not None)
        self.btn_phase3.setEnabled(self.db_conn is not None)
        
        # Change color based on status
        if value == 100:
            self.status_label.setStyleSheet("color: #27AE60;")
        elif value == 0 and "Error" in message:
            self.status_label.setStyleSheet("color: #E74C3C;")
        else:
            self.status_label.setStyleSheet("color: #2C3E50;")

    def closeEvent(self, event):
        """Handle window close with cleanup"""
        if self.db_conn:
            self.db_conn.close()
        if self.video_capture:
            self.video_capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = SurfAnalysisApp()
    window.show()
    
    sys.exit(app.exec_())