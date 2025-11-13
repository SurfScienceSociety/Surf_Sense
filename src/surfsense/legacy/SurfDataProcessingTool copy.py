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
                             QLabel, QComboBox, QPushButton, QFileDialog, QProgressBar, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Sensor variables
SENSOR_VARIABLES = {
    "FFS": ["X_adc", "Y_adc", "X_load", "Y_load", "Res_Force", "X_Flow", "Y_Flow", "Res_Flow", "Res_Angle", "Temperature"],
    "IMU": ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mag_X", "Mag_Y", "Mag_Z", "Yaw", "Pitch", "Roll"],
    "$GNRMC": ["Speed_Knots", "Angle"]
}

class SurfAnalysisApp(QMainWindow):
    update_progress = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surf Analysis Suite")
        self.resize(1400, 900)

        # Application state
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.abspath(os.path.join(self.base_dir, 'data'))
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_conn = None
        self.video_capture = None
        self.event_data = pd.DataFrame()
        self.current_event = None

        # Initialize UI
        self.init_ui()
        self.init_connections()

    def init_ui(self):
        """Initialize the modern user interface"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Panel: Processing Pipeline
        self.control_panel = QGroupBox("Processing Pipeline")
        self.control_layout = QVBoxLayout()

        self.btn_phase0 = QPushButton("1. Preprocess Raw Data")
        self.btn_phase1 = QPushButton("2. Create Database")
        self.btn_phase2 = QPushButton("3. Clean Data")
        self.btn_phase3 = QPushButton("4. Visualize Data")

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.control_layout.addWidget(self.btn_phase0)
        self.control_layout.addWidget(self.btn_phase1)
        self.control_layout.addWidget(self.btn_phase2)
        self.control_layout.addWidget(self.btn_phase3)
        self.control_layout.addWidget(self.progress_bar)
        self.control_layout.addWidget(self.status_label)
        self.control_layout.addStretch()
        self.control_panel.setLayout(self.control_layout)

        # Right Panel: Visualization
        self.viz_panel = QWidget()
        self.viz_layout = QVBoxLayout()

        self.viz_controls = QGroupBox("Data Visualization")
        self.viz_control_layout = QGridLayout()

        self.event_label = QLabel("Event:")
        self.event_combo = QComboBox()

        self.sensor_label = QLabel("Sensor:")
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(SENSOR_VARIABLES.keys())

        self.var_label = QLabel("Variable:")
        self.var_combo = QComboBox()

        self.btn_load_video = QPushButton("Load Video")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")

        self.viz_control_layout.addWidget(self.event_label, 0, 0)
        self.viz_control_layout.addWidget(self.event_combo, 0, 1, 1, 2)
        self.viz_control_layout.addWidget(self.sensor_label, 1, 0)
        self.viz_control_layout.addWidget(self.sensor_combo, 1, 1)
        self.viz_control_layout.addWidget(self.var_label, 1, 2)
        self.viz_control_layout.addWidget(self.var_combo, 1, 3)
        self.viz_control_layout.addWidget(self.btn_load_video, 2, 0)
        self.viz_control_layout.addWidget(self.btn_play, 2, 1)
        self.viz_control_layout.addWidget(self.btn_pause, 2, 2)

        self.viz_controls.setLayout(self.viz_control_layout)

        self.figure, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(400, 300)

        self.viz_layout.addWidget(self.viz_controls)
        self.viz_layout.addWidget(self.toolbar)
        self.viz_layout.addWidget(self.canvas)
        self.viz_layout.addWidget(self.video_display)

        self.main_layout.addWidget(self.control_panel, 1)
        self.main_layout.addWidget(self.viz_panel, 3)

        self.playback_timer = QTimer()
        self.playback_timer.setInterval(33)  # ~30fps

        self.apply_styles()
        self.update_variable_combo()

    def apply_styles(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F0F2F5;
            }
            QGroupBox {
                font: bold 12px 'Arial';
                border: 1px solid #D3D3D3;
                border-radius: 8px;
                margin-top: 15px;
                padding: 10px;
                background-color: #FFFFFF;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font: 11px 'Arial';
            }
            QPushButton:hover {
                background-color: #005BB5;
            }
            QPushButton:disabled {
                background-color: #A0A0A0;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #D3D3D3;
                border-radius: 4px;
                background-color: #FFFFFF;
            }
            QLabel {
                color: #333333;
                font: 11px 'Arial';
            }
            QProgressBar {
                border: 1px solid #D3D3D3;
                border-radius: 4px;
                text-align: center;
                background-color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
            }
        """)
        font = QFont("Arial", 10)
        self.setFont(font)
        plt.style.use('ggplot')  # Usando 'ggplot' em vez de 'seaborn'
        self.figure.patch.set_facecolor('#F0F2F5')
        self.ax.set_facecolor('#FFFFFF')

    def init_connections(self):
        """Connect signals and slots"""
        self.btn_phase0.clicked.connect(self.run_phase0)
        self.btn_phase1.clicked.connect(self.run_phase1)
        self.btn_phase2.clicked.connect(self.run_phase2)
        self.btn_phase3.clicked.connect(self.run_phase3)
        self.sensor_combo.currentTextChanged.connect(self.update_variable_combo)
        self.event_combo.currentTextChanged.connect(self.load_event_data)
        self.var_combo.currentTextChanged.connect(self.update_plot)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_play.clicked.connect(self.start_playback)
        self.btn_pause.clicked.connect(self.pause_playback)
        self.playback_timer.timeout.connect(self.update_video_frame)
        self.update_progress.connect(self.update_progress_display)

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

            self.update_progress.emit(100, "Phase 0 complete.")
        except Exception as e:
            self.update_progress.emit(0, f"Error in Phase 0: {str(e)}")

    def filter_data(self, file_path):
        """Filter valid data lines"""
        filtered = []
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if (row[0].startswith("FFS") and len(row) == 11) or \
                   (row[0].startswith("IMU") and len(row) == 13) or \
                   (row[0].startswith("$GNRMC") and len(row) == 13):
                    filtered.append(row)
        return filtered

    def add_timestamps(self, file_path):
        """Add timestamps between GPS data"""
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        new_lines = []
        current_time = None
        ffs_imu_pairs = []

        for line in lines:
            if self.is_valid_gnrmc(line):
                parts = line.split(',')
                if current_time and ffs_imu_pairs:
                    next_time = self.parse_time(parts[1])
                    time_diff = next_time - current_time
                    interval = time_diff / (len(ffs_imu_pairs) + 1) if ffs_imu_pairs else timedelta()
                    for i, (ffs_line, imu_line) in enumerate(ffs_imu_pairs):
                        timestamp = current_time + (i + 1) * interval
                        timestamp_str = self.format_time(timestamp)
                        new_lines.append(f"{timestamp_str},{ffs_line}")
                        new_lines.append(f"{timestamp_str},{imu_line}")
                    ffs_imu_pairs = []
                current_time = self.parse_time(parts[1])
                timestamp_str = self.format_time(current_time)
                new_lines.append(f"{timestamp_str},{line}")
            elif line.startswith('FFS'):
                ffs_line = line
            elif line.startswith('IMU'):
                imu_line = line
                if 'ffs_line' in locals():
                    ffs_imu_pairs.append((ffs_line, imu_line))

        output_path = self.force_data_folder(file_path, 'timestamped')
        with open(output_path, 'w') as f:
            f.write("\n".join(new_lines) + "\n")
        return output_path

    def reorder_timestamps(self, file_path):
        """Reorganize timestamps"""
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        new_lines = []
        for line in lines:
            parts = line.split(',')
            if len(parts) > 1 and re.match(r'\d{2}:\d{2}:\d{2}\.\d{3}', parts[0]):
                time_part, millis_part = parts[0].split('.')
                new_line = f"{time_part},{millis_part}," + ",".join(parts[1:])
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        
        output_path = self.force_data_folder(file_path, 'reordered')
        with open(output_path, 'w') as f:
            f.write("\n".join(new_lines) + "\n")
        return output_path

    def trim_to_valid_gps(self, file_path):
        """Trim file to first valid GPS message"""
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        start_index = None
        for i, line in enumerate(lines):
            parts = line.split(',')
            if len(parts) >= 5 and parts[2].startswith('$GNRMC') and parts[4] == 'A':
                start_index = i
                break
        
        output_path = self.force_data_folder(file_path, 'final')
        if start_index is not None:
            final_lines = lines[start_index:]
            with open(output_path, 'w') as f:
                f.write("\n".join(final_lines) + "\n")
        else:
            shutil.copy(file_path, output_path)
        return output_path

    # === Phase 1: Database Creation ===
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
            db_path = self.select_save_file("Save Database As", os.path.join(self.data_dir, "surf_data.db"), "Database Files (*.db)")
            if not db_path:
                self.update_progress.emit(0, "No database location selected.")
                return

            self.update_progress.emit(40, "Processing data...")
            self.process_database(self.final_file, trim_file, db_path)
            self.db_path = db_path
            self.db_conn = sqlite3.connect(db_path)
            self.update_progress.emit(100, "Phase 1 complete: Database created.")
        except Exception as e:
            self.update_progress.emit(0, f"Error in Phase 1: {str(e)}")

    def process_database(self, dataset_path, trimming_path, db_path):
        """Process trimming and create database"""
        trimming_df = pd.read_csv(trimming_path)
        homogenized_file = self.homogenize_csv(dataset_path)
        dataset_df = pd.read_csv(homogenized_file, header=None, dtype=str, low_memory=False)

        num_cols = dataset_df.shape[1]
        base_columns = ['Time', 'Millisec', 'Data_Type']
        if num_cols == 4:
            dataset_df.columns = base_columns + ['Data']
        else:
            dataset_df.columns = base_columns + [f'col_{i}' for i in range(3, num_cols)]
            dataset_df['Data'] = dataset_df.iloc[:, 3:].apply(lambda row: ','.join(str(val) for val in row), axis=1)
            dataset_df = dataset_df[base_columns + ['Data']]

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS Events")
        cursor.execute('''
            CREATE TABLE Events (
                Event_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Event_Name TEXT,
                Source TEXT,
                Start TEXT,
                Stop TEXT,
                Start2 TEXT,
                Stop2 TEXT,
                Duration TEXT,
                Annotations TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE Sensor_Data (
                Data_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Event_ID INTEGER,
                Time TEXT,
                Millisec TEXT,
                Data_Type TEXT,
                Data TEXT,
                FOREIGN KEY (Event_ID) REFERENCES Events (Event_ID)
            )
        ''')

        for _, row in trimming_df.iterrows():
            event_name = row['Events']
            source = row['Source']
            start = row['Start']
            stop = row['Stop']
            start2 = row['Start2']
            stop2 = row['Stop2']
            duration = row['Duration']
            annotations = row['Annotations'] if not pd.isna(row['Annotations']) else ""

            if pd.isna(start2) or pd.isna(stop2):
                continue

            cursor.execute('''
                INSERT INTO Events (Event_Name, Source, Start, Stop, Start2, Stop2, Duration, Annotations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (event_name, source, start, stop, start2, stop2, duration, annotations))

            event_id = cursor.lastrowid
            mask = (dataset_df['Time'] >= start2) & (dataset_df['Time'] <= stop2)
            filtered_df = dataset_df.loc[mask]

            for _, data_row in filtered_df.iterrows():
                cursor.execute('''
                    INSERT INTO Sensor_Data (Event_ID, Time, Millisec, Data_Type, Data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (event_id, data_row['Time'], data_row['Millisec'], data_row['Data_Type'], data_row['Data']))

        conn.commit()
        conn.close()

    # === Phase 2: Data Cleaning ===
    def run_phase2(self):
        """Clean the database"""
        if not self.db_conn:
            self.update_progress.emit(0, "Error: Run Phase 1 first.")
            return

        try:
            self.update_progress.emit(0, "Starting data cleaning...")
            self.clean_data()
            self.update_progress.emit(100, "Phase 2 complete: Data cleaned.")
            self.load_events()
        except Exception as e:
            self.update_progress.emit(0, f"Error in Phase 2: {str(e)}")

    def clean_data(self):
        """Clean data by removing outliers and interpolating"""
        db_clean_path = self.db_path.replace('.db', '_cleaned.db')
        conn_clean = sqlite3.connect(db_clean_path)
        cursor_clean = conn_clean.cursor()

        cursor_clean.execute('''
            CREATE TABLE Events (
                Event_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Event_Name TEXT,
                Source TEXT,
                Start TEXT,
                Stop TEXT,
                Start2 TEXT,
                Stop2 TEXT,
                Duration TEXT,
                Annotations TEXT
            )
        ''')
        cursor_clean.execute('''
            CREATE TABLE Sensor_Data (
                Data_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Event_ID INTEGER,
                Time TEXT,
                Millisec TEXT,
                Data_Type TEXT,
                Data TEXT,
                FOREIGN KEY (Event_ID) REFERENCES Events (Event_ID)
            )
        ''')

        events_df = pd.read_sql_query("SELECT * FROM Events", self.db_conn)
        events_df.to_sql('Events', conn_clean, if_exists='append', index=False)

        for _, event in events_df.iterrows():
            event_id = event['Event_ID']
            df = pd.read_sql_query("SELECT * FROM Sensor_Data WHERE Event_ID = ?", self.db_conn, params=(event_id,))

            if df.empty:
                continue

            df_gnrmc = df[df['Data_Type'] == '$GNRMC'].copy()
            df_other = df[df['Data_Type'] != '$GNRMC'].copy()

            if not df_other.empty:
                data_expanded = df_other['Data'].str.split(',', expand=True).apply(pd.to_numeric, errors='coerce')
                for col in data_expanded.columns:
                    col_data = data_expanded[col]
                    mean = col_data.mean()
                    std = col_data.std()
                    outliers_mask = (col_data - mean).abs() > 3 * std
                    col_data[outliers_mask] = np.nan
                    col_data.interpolate(method='linear', inplace=True, limit_direction='both')
                    data_expanded[col] = col_data
                df_other['Data'] = data_expanded.apply(lambda row: ','.join(row.astype(str)), axis=1)

            df_clean = pd.concat([df_other, df_gnrmc]).sort_index()
            df_clean = df_clean.drop(columns=['Data_ID'])
            df_clean.to_sql('Sensor_Data', conn_clean, if_exists='append', index=False)

        conn_clean.commit()
        conn_clean.close()
        self.db_conn.close()
        self.db_conn = sqlite3.connect(db_clean_path)
        self.db_path = db_clean_path

    # === Phase 3: Visualization ===
    def run_phase3(self):
        """Load visualization"""
        if not self.db_conn:
            self.update_progress.emit(0, "Error: Run Phase 2 first.")
            return
        self.update_progress.emit(100, "Phase 3 ready: Visualization loaded.")
        self.load_events()

    def load_events(self):
        """Load events into combo box"""
        if not self.db_conn:
            return
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT Event_Name FROM Events")
        events = cursor.fetchall()
        self.event_combo.clear()
        for (event_name,) in events:
            event_type = "Waves" if "Wave" in event_name else "Paddlings"
            self.event_combo.addItem(f"{event_type}: {event_name}")

    def load_event_data(self):
        """Load data for selected event"""
        if not self.db_conn:
            return

        current_text = self.event_combo.currentText()
        if not current_text or ": " not in current_text:
            return

        event_name = current_text.split(": ")[1]
        event = pd.read_sql_query("SELECT * FROM Events WHERE Event_Name = ?", self.db_conn, params=(event_name,)).iloc[0]
        self.current_event = {'name': event['Event_Name'], 'start': event['Start'], 'stop': event['Stop']}

        df = pd.read_sql_query("SELECT * FROM Sensor_Data WHERE Event_ID = (SELECT Event_ID FROM Events WHERE Event_Name = ?)",
                               self.db_conn, params=(event_name,))
        df['Millisec'] = df['Millisec'].fillna(0).astype(int).astype(str).str.zfill(3)
        df['Timestamp'] = pd.to_datetime(df['Time'] + '.' + df['Millisec'], format='%H:%M:%S.%f', errors='coerce')
        self.event_data = df[df['Data_Type'] == self.sensor_combo.currentText()]
        self.update_plot()

    def update_variable_combo(self):
        """Update variable selector"""
        self.var_combo.clear()
        sensor_type = self.sensor_combo.currentText()
        self.var_combo.addItems(SENSOR_VARIABLES[sensor_type])
        self.update_plot()

    def update_plot(self):
        """Update plot with selected data"""
        if self.event_data.empty or self.var_combo.currentIndex() == -1:
            return

        self.ax.clear()
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

        self.ax.plot(plot_data['Timestamp'], plot_data['Value'], color='dodgerblue', linewidth=2)
        self.ax.set_title(f"{self.current_event['name']} - {variable}", fontsize=14)
        self.ax.set_xlabel("Time", fontsize=12)
        self.ax.set_ylabel(variable, fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        self.figure.tight_layout()
        self.canvas.draw()

    # === Video Methods ===
    def load_video(self):
        """Load video file"""
        video_path = self.select_file("Select Video File", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            if self.video_capture:
                self.video_capture.release()
            self.video_capture = cv2.VideoCapture(video_path)
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.playback_timer.setInterval(int(1000 / self.video_fps))
            ret, frame = self.video_capture.read()
            if ret:
                self.display_video_frame(frame)

    def start_playback(self):
        """Start video playback"""
        if self.video_capture and self.current_event:
            start_time = self.time_to_seconds(self.current_event['start'])
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            self.playback_timer.start()

    def pause_playback(self):
        """Pause video playback"""
        self.playback_timer.stop()

    def update_video_frame(self):
        """Update video frame"""
        if not self.video_capture:
            self.playback_timer.stop()
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.playback_timer.stop()
            return

        self.display_video_frame(frame)
        current_pos = self.video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        start_time = self.time_to_seconds(self.current_event['start'])
        event_time = self.event_data['Timestamp'].iloc[0] + pd.Timedelta(seconds=current_pos - start_time)
        if hasattr(self, 'cursor_line'):
            self.cursor_line.set_xdata([event_time])
            self.canvas.draw()

    def display_video_frame(self, frame):
        """Display video frame"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        target_w, target_h = self.video_display.width(), self.video_display.height()
        aspect = w / h
        if w > target_w or h > target_h:
            if (target_w / aspect) <= target_h:
                w, h = target_w, int(target_w / aspect)
            else:
                h, w = target_h, int(target_h * aspect)
        frame = cv2.resize(frame, (w, h))
        q_img = QImage(frame, w, h, frame.strides[0], QImage.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(q_img))

    # === Helper Methods ===
    def select_file(self, title, directory, file_filter):
        """Select a file"""
        file_path, _ = QFileDialog.getOpenFileName(self, title, directory, file_filter)
        return file_path

    def select_save_file(self, title, default_name, file_filter):
        """Select save location"""
        file_path, _ = QFileDialog.getSaveFileName(self, title, default_name, file_filter)
        return file_path

    def force_data_folder(self, file_path, prefix):
        """Generate file path in data directory"""
        filename = os.path.basename(file_path)
        if not filename.endswith('.csv'):
            filename += '.csv'
        parts = filename.split('_', 1)
        new_name = f"{prefix}_{parts[1]}" if len(parts) > 1 else f"{prefix}_{filename}"
        return os.path.join(self.data_dir, new_name)

    def homogenize_csv(self, file_path):
        """Homogenize CSV file"""
        with open(file_path, 'r') as f:
            lines = [line.strip().split(',') for line in f if line.strip()]
        max_columns = max(len(line) for line in lines)
        homogenized_lines = [line + [''] * (max_columns - len(line)) for line in lines]
        output_file = os.path.join(self.data_dir, 'homogenized_temp.csv')
        with open(output_file, 'w') as f:
            for line in homogenized_lines:
                f.write(','.join(line) + '\n')
        return output_file

    def is_valid_gnrmc(self, line):
        """Check if GPS line is valid"""
        if not line.startswith('$GNRMC'):
            return False
        match = re.match(r'^\$(.*)\*(\w\w)$', line.strip())
        if not match:
            return False
        data, checksum = match.groups()
        calculated_checksum = 0
        for char in data:
            calculated_checksum ^= ord(char)
        return f'{calculated_checksum:02X}' == checksum.upper()

    def parse_time(self, time_str):
        """Parse time string"""
        return datetime.strptime(time_str, '%H%M%S.%f')

    def format_time(self, dt):
        """Format datetime to string"""
        return dt.strftime('%H:%M:%S.%f')[:-3]

    def parse_gnrmc(self, data_str):
        """Parse $GNRMC data"""
        try:
            if not data_str.startswith('$GNRMC'):
                return None
            parts = data_str.split(',')
            if len(parts) < 9:
                return None
            speed = float(parts[7]) if parts[7] else 0.0
            angle = float(parts[8]) if parts[8] else 0.0
            return {'Speed_Knots': speed, 'Angle': angle}
        except (ValueError, IndexError):
            return None

    def time_to_seconds(self, time_str):
        """Convert time string to seconds"""
        parts = list(map(int, str(time_str).split(':')))
        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    def update_progress_display(self, value, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.btn_phase1.setEnabled(hasattr(self, 'final_file'))
        self.btn_phase2.setEnabled(self.db_conn is not None)
        self.btn_phase3.setEnabled(self.db_conn is not None)

    def closeEvent(self, event):
        """Handle window close"""
        if self.db_conn:
            self.db_conn.close()
        if self.video_capture:
            self.video_capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SurfAnalysisApp()
    window.show()
    sys.exit(app.exec_())