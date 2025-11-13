import sys
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QListWidget, QLabel, QFileDialog
)

# Função para seleção de ficheiro com filtro
def select_file(file_type, file_extension):
    dialog = QFileDialog()
    file_path, _ = dialog.getOpenFileName(None, f'Selecione o ficheiro {file_type}', '', file_extension)
    return file_path

# Função para leitura da base de dados
def read_database(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM Sensor_Data WHERE Data_Type = '$GNRMC'", conn)
    conn.close()

    # Preparação do timestamp
    df['Millisec'] = df['Millisec'].astype(float).astype(int).astype(str).str.zfill(3)
    df['Timestamp'] = pd.to_datetime(df['Time'] + '.' + df['Millisec'], format='%H:%M:%S.%f')

    # Separar os dados da coluna Data (valores já numéricos)
    data_split = df['Data'].str.split(',', expand=True)
    df['Longitude'] = pd.to_numeric(data_split[0], errors='coerce')
    df['Latitude'] = pd.to_numeric(data_split[1], errors='coerce')

    # Opcional: velocidade se disponível
    if data_split.shape[1] > 2:
        df['Speed'] = pd.to_numeric(data_split[2], errors='coerce')
    else:
        df['Speed'] = 0.0

    return df

# Função para leitura do trimming
def read_trimming(trim_path):
    return pd.read_csv(trim_path)

# Função para filtrar dados da base conforme evento selecionado
def filter_event_data(df, event):
    start_time = pd.to_datetime(event['Start2'])
    end_time = pd.to_datetime(event['Stop2'])
    event_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]
    return event_df

# Função para plotagem estática
def plot_static(event_df):
    plt.figure(figsize=(10, 6))
    plt.plot(event_df['Longitude'], event_df['Latitude'], marker='o')
    plt.title('Trajeto Estático do Evento')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

# Função para reprodução temporal
def temporal_replay(event_df):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Reprodução Temporal do Evento')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)

    # Trajeto completo como fundo
    ax.plot(event_df['Longitude'], event_df['Latitude'], linestyle='--', alpha=0.5)

    # Ponto móvel
    point, = ax.plot([], [], marker='o', color='red')

    last_time = None

    for _, row in event_df.iterrows():
        current_time = row['Timestamp']
        if last_time is not None:
            delta = (current_time - last_time).total_seconds()
            if delta > 0:
                time.sleep(delta)
        point.set_data(row['Longitude'], row['Latitude'])
        plt.pause(0.001)
        last_time = current_time

    plt.ioff()
    plt.show()

# Classe principal da aplicação
class GPSVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GPS Visualizer - PyQt5')
        self.setGeometry(100, 100, 600, 400)

        self.layout = QVBoxLayout()

        self.label = QLabel('Selecione o evento de interesse:')
        self.layout.addWidget(self.label)

        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        self.load_button = QPushButton('Carregar Ficheiros')
        self.load_button.clicked.connect(self.load_files)
        self.layout.addWidget(self.load_button)

        self.static_button = QPushButton('Plotagem Estática')
        self.static_button.clicked.connect(self.plot_static_event)
        self.static_button.setEnabled(False)
        self.layout.addWidget(self.static_button)

        self.replay_button = QPushButton('Reprodução Temporal')
        self.replay_button.clicked.connect(self.temporal_replay_event)
        self.replay_button.setEnabled(False)
        self.layout.addWidget(self.replay_button)

        self.setLayout(self.layout)

        self.df = None
        self.trimming_df = None
        self.selected_event = None

    def load_files(self):
        db_path = select_file('DB', 'Database Files (*.db)')
        trim_path = select_file('Trimming CSV', 'CSV Files (*.csv)')

        if db_path and trim_path:
            self.df = read_database(db_path)
            self.trimming_df = read_trimming(trim_path)

            self.list_widget.clear()
            for idx, row in self.trimming_df.iterrows():
                self.list_widget.addItem(f"{idx}: {row['Events']} | {row['Start2']} -> {row['Stop2']} | {row['Annotations']}")

            self.list_widget.currentRowChanged.connect(self.on_event_selected)

    def on_event_selected(self, index):
        if index >= 0:
            self.selected_event = self.trimming_df.iloc[index]
            self.static_button.setEnabled(True)
            self.replay_button.setEnabled(True)

    def plot_static_event(self):
        if self.selected_event is not None:
            event_df = filter_event_data(self.df, self.selected_event)
            plot_static(event_df)

    def temporal_replay_event(self):
        if self.selected_event is not None:
            event_df = filter_event_data(self.df, self.selected_event)
            temporal_replay(event_df)

# Execução principal
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GPSVisualizer()
    window.show()
    sys.exit(app.exec_())
