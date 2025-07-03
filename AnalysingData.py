import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QHBoxLayout, QVBoxLayout, QTabWidget, QStackedWidget,
                             QGroupBox, QFrame, QSizePolicy, QScrollArea, QComboBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap
from scipy.fft import fft, fftfreq
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import learning_curve
from scipy.signal import spectrogram, hilbert, stft
from scipy.stats import kurtosis

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GearHealthMonitoringSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fan Condition Monitoring and Evaluation System")
        self.setGeometry(100, 100, 1600, 1000)
        self.setWindowIcon(QIcon("icons/gear.png"))

        # Initialize variables
        self.loaded_data = None
        self.cart_model = None
        self.id3_model = None
        self.current_model = None
        self.current_sample = None
        self.class_labels = None
        self.sampling_rate = 10000
        self.normal_sample = None  # Normal state sample
        self.fault_sample = None   # Fault state sample

        # Load pre-trained models
        self.load_models()

        # Create UI
        self.init_ui()
        self.apply_styles()

    def load_models(self):
        """Load pre-trained decision tree models"""
        cart_model_path = 'result/decision_tree_model_cart.joblib'
        if os.path.exists(cart_model_path):
            try:
                self.cart_model = joblib.load(cart_model_path)
                if hasattr(self.cart_model, 'classes_'):
                    self.class_labels = [str(label) for label in self.cart_model.classes_]
                else:
                    self.class_labels = ["Class 1", "Class 2", "Class 3", "Class 4"]
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"CART model loading failed: {str(e)}")
                self.cart_model = None

        id3_model_path = 'result/decision_tree_model_id3.joblib'
        if os.path.exists(id3_model_path):
            try:
                self.id3_model = joblib.load(id3_model_path)
                if not self.class_labels and hasattr(self.id3_model, 'classes_'):
                    self.class_labels = [str(label) for label in self.id3_model.classes_]
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"ID3 model loading failed: {str(e)}")
                self.id3_model = None

        self.current_model = self.cart_model

    def init_ui(self):
        """Initialize professional UI interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        self.create_navigation_bar(main_layout)
        self.create_content_area(main_layout)

    def create_navigation_bar(self, parent_layout):
        """Create professional navigation bar"""
        nav_bar = QFrame()
        nav_bar.setFixedWidth(220)
        nav_bar.setObjectName("navBar")

        nav_layout = QVBoxLayout()
        nav_bar.setLayout(nav_layout)

        logo_label = QLabel()
        logo_pixmap = QPixmap("icons/gear_diagnosis.png").scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel("Fan Condition\nMonitoring and\nEvaluation System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: orange;")

        nav_layout.addWidget(logo_label)
        nav_layout.addWidget(title_label)
        nav_layout.addSpacing(20)

        self.nav_buttons = []
        nav_items = [
            ("Data Import", "icons/import.png"),
            ("Time Domain Analysis", "icons/time.png"),
            ("Feature Indicators", "icons/features.png"),
            ("Spectral Kurtosis", "icons/kurtosis.png"),
            ("Confusion Matrix", "icons/matrix.png"),
            ("Learning Curve", "icons/curve.png"),
            ("Decision Tree", "icons/tree.png"),
            ("Fault Prediction", "icons/predict.png"),
            ("Exit System", "icons/exit.png")
        ]

        for text, icon in nav_items:
            btn = QPushButton(text)
            btn.setIcon(QIcon(icon))
            btn.setIconSize(QSize(24, 24))
            btn.setCheckable(True)
            btn.setObjectName("navButton")
            btn.setMinimumHeight(50)
            btn_font = QFont()
            btn_font.setPointSize(16)
            btn.setFont(btn_font)
            if text == "Fault Prediction":
                btn.clicked.connect(self.predict_fault)
            elif text == "Exit System":
                btn.clicked.connect(self.close)
            else:
                btn.clicked.connect(lambda _, idx=len(self.nav_buttons): self.content_stack.setCurrentIndex(idx))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        if self.nav_buttons:
            self.nav_buttons[0].setChecked(True)

        nav_layout.addStretch()
        parent_layout.addWidget(nav_bar)

    def create_content_area(self, parent_layout):
        """Create content area"""
        content_frame = QFrame()
        content_frame.setObjectName("contentFrame")

        content_layout = QVBoxLayout()
        content_frame.setLayout(content_layout)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("CART Algorithm")
        self.algorithm_combo.addItem("ID3 Algorithm")
        self.algorithm_combo.setCurrentIndex(0)
        self.algorithm_combo.currentIndexChanged.connect(self.switch_algorithm)
        self.algorithm_combo.hide()
        content_layout.addWidget(self.algorithm_combo, alignment=Qt.AlignRight)

        # Create stacked widget
        self.content_stack = QStackedWidget()
        self.content_stack.currentChanged.connect(self.on_page_changed)

        self.create_import_page()
        self.create_time_domain_page()
        self.create_features_page()
        self.create_kurtosis_page()
        self.create_confusion_matrix_page()
        self.create_learning_curve_page()
        self.create_decision_tree_page()

        content_layout.addWidget(self.content_stack)
        parent_layout.addWidget(content_frame)

    def on_page_changed(self, index):
        """Handle page changes"""
        if index >= 4:
            self.algorithm_combo.show()
        else:
            self.algorithm_combo.hide()

    def switch_algorithm(self, index):
        """Switch current algorithm"""
        if index == 0:
            self.current_model = self.cart_model
        else:
            self.current_model = self.id3_model

        if self.content_stack.currentIndex() >= 4:
            self.update_model_dependent_displays()

    def update_model_dependent_displays(self):
        """Update model-dependent displays"""
        current_index = self.content_stack.currentIndex()
        if current_index == 4:
            self.show_confusion_matrix()
        elif current_index == 5:
            self.show_learning_curve()
        elif current_index == 6:
            self.show_decision_tree()

    def create_import_page(self):
        """Data import page"""
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("Data Import")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)

        import_btn = QPushButton("Import Normal Data")
        import_btn.setIcon(QIcon("icons/import.png"))
        import_btn.setObjectName("actionButton")
        import_btn.clicked.connect(lambda: self.import_data('normal'))
        import_btn_font = QFont()
        import_btn_font.setPointSize(16)
        import_btn.setFont(import_btn_font)

        import_fault_btn = QPushButton("Import Fault Data")
        import_fault_btn.setIcon(QIcon("icons/import.png"))
        import_fault_btn.setObjectName("actionButton")
        import_fault_btn.clicked.connect(lambda: self.import_data('fault'))
        import_fault_btn_font = QFont()
        import_fault_btn_font.setPointSize(16)
        import_fault_btn.setFont(import_fault_btn_font)

        self.data_status_label = QLabel("Waiting for data import...")
        self.data_status_label.setAlignment(Qt.AlignCenter)
        data_status_font = QFont()
        data_status_font.setPointSize(16)
        self.data_status_label.setFont(data_status_font)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(import_btn, alignment=Qt.AlignCenter)
        layout.addWidget(import_fault_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.data_status_label)
        layout.addStretch()

        self.content_stack.addWidget(page)

    def create_time_domain_page(self):
        """Time Domain Analysis Page"""
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("Time Domain Analysis")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)

        self.time_figure = plt.figure(figsize=(12, 8), facecolor='none')
        self.time_canvas = FigureCanvas(self.time_figure)

        analyze_btn = QPushButton("Analyze Signal")
        analyze_btn.setIcon(QIcon("icons/analyze.png"))
        analyze_btn.setObjectName("actionButton")
        analyze_btn.clicked.connect(self.show_time_domain_analysis)
        analyze_btn_font = QFont()
        analyze_btn_font.setPointSize(16)
        analyze_btn.setFont(analyze_btn_font)

        self.time_prediction_label = QLabel("Prediction results will be displayed here")
        self.time_prediction_label.setAlignment(Qt.AlignCenter)
        prediction_font = QFont()
        prediction_font.setPointSize(16)
        self.time_prediction_label.setFont(prediction_font)
        self.time_prediction_label.setStyleSheet("color: red;")

        layout.addWidget(title)
        layout.addWidget(self.time_canvas)
        layout.addWidget(analyze_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.time_prediction_label)

        self.content_stack.addWidget(page)

    def create_features_page(self):
        """Feature Indicators Page"""
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("Feature Indicator Analysis")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)

        self.features_figure = plt.figure(figsize=(12, 6), facecolor='none')
        self.features_canvas = FigureCanvas(self.features_figure)

        analyze_btn = QPushButton("Calculate Features")
        analyze_btn.setIcon(QIcon("icons/calculate.png"))
        analyze_btn.setObjectName("actionButton")
        analyze_btn.clicked.connect(self.show_features_analysis)
        analyze_btn_font = QFont()
        analyze_btn_font.setPointSize(16)
        analyze_btn.setFont(analyze_btn_font)

        self.features_prediction_label = QLabel("Prediction results will be displayed here")
        self.features_prediction_label.setAlignment(Qt.AlignCenter)
        prediction_font = QFont()
        prediction_font.setPointSize(16)
        self.features_prediction_label.setFont(prediction_font)
        self.features_prediction_label.setStyleSheet("color: red;")

        layout.addWidget(title)
        layout.addWidget(self.features_canvas)
        layout.addWidget(analyze_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.features_prediction_label)

        self.content_stack.addWidget(page)

    def create_kurtosis_page(self):
        """Spectral Kurtosis Analysis Page"""
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("Spectral Kurtosis Analysis")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)

        self.kurtosis_figure = plt.figure(figsize=(12, 6), facecolor='none')
        self.kurtosis_canvas = FigureCanvas(self.kurtosis_figure)

        analyze_btn = QPushButton("Analyze Spectral Kurtosis")
        analyze_btn.setIcon(QIcon("icons/analyze.png"))
        analyze_btn.setObjectName("actionButton")
        analyze_btn.clicked.connect(self.show_spectral_kurtosis)
        analyze_btn_font = QFont()
        analyze_btn_font.setPointSize(16)
        analyze_btn.setFont(analyze_btn_font)

        self.kurtosis_prediction_label = QLabel("Prediction results will be displayed here")
        self.kurtosis_prediction_label.setAlignment(Qt.AlignCenter)
        prediction_font = QFont()
        prediction_font.setPointSize(16)
        self.kurtosis_prediction_label.setFont(prediction_font)
        self.kurtosis_prediction_label.setStyleSheet("color: red;")

        layout.addWidget(title)
        layout.addWidget(self.kurtosis_canvas)
        layout.addWidget(analyze_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.kurtosis_prediction_label)

        self.content_stack.addWidget(page)

    def create_confusion_matrix_page(self):
        """Confusion Matrix Page"""
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("Confusion Matrix")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)

        self.confusion_figure = plt.figure(figsize=(16, 12), facecolor='none', dpi=100)
        self.confusion_canvas = FigureCanvas(self.confusion_figure)

        layout.addWidget(self.confusion_canvas, alignment=Qt.AlignCenter)

        load_btn = QPushButton("Display Confusion Matrix")
        load_btn.setIcon(QIcon("icons/load.png"))
        load_btn.setObjectName("actionButton")
        load_btn.clicked.connect(self.show_confusion_matrix)
        load_btn_font = QFont()
        load_btn_font.setPointSize(16)
        load_btn.setFont(load_btn_font)

        layout.addWidget(title)
        layout.addWidget(self.confusion_canvas)
        layout.addWidget(load_btn, alignment=Qt.AlignCenter)

        self.content_stack.addWidget(page)

    def create_learning_curve_page(self):
        """Learning Curve Page"""
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("Learning Curve")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)

        self.learning_curve_figure = plt.figure(figsize=(16, 10), facecolor='none', dpi=100)
        self.learning_curve_canvas = FigureCanvas(self.learning_curve_figure)

        layout.addWidget(self.learning_curve_canvas, alignment=Qt.AlignCenter)

        button_layout = QHBoxLayout()

        load_btn = QPushButton("Display Learning Curve")
        load_btn.setIcon(QIcon("icons/load.png"))
        load_btn.setObjectName("actionButton")
        load_btn.clicked.connect(self.show_learning_curve)
        load_btn_font = QFont()
        load_btn_font.setPointSize(16)
        load_btn.setFont(load_btn_font)

        show_acc_btn = QPushButton("Show Accuracy")
        show_acc_btn.setIcon(QIcon("icons/accuracy.png"))
        show_acc_btn.setObjectName("actionButton")
        show_acc_btn.clicked.connect(self.show_accuracy)
        show_acc_btn_font = QFont()
        show_acc_btn_font.setPointSize(16)
        show_acc_btn.setFont(show_acc_btn_font)

        button_layout.addWidget(load_btn)
        button_layout.addWidget(show_acc_btn)

        self.accuracy_label = QLabel("Accuracy will be displayed here")
        self.accuracy_label.setAlignment(Qt.AlignCenter)
        accuracy_font = QFont()
        accuracy_font.setPointSize(18)
        self.accuracy_label.setFont(accuracy_font)

        layout.addWidget(title)
        layout.addWidget(self.learning_curve_canvas)
        layout.addLayout(button_layout)
        layout.addWidget(self.accuracy_label)

        self.content_stack.addWidget(page)

    def create_decision_tree_page(self):
        """Decision Tree Visualization Page"""
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("Decision Tree Visualization")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)

        self.tree_figure = plt.figure(figsize=(16, 10), facecolor='none')
        self.tree_canvas = FigureCanvas(self.tree_figure)

        load_btn = QPushButton("Display Decision Tree")
        load_btn.setIcon(QIcon("icons/load.png"))
        load_btn.setObjectName("actionButton")
        load_btn.clicked.connect(self.show_decision_tree)
        load_btn_font = QFont()
        load_btn_font.setPointSize(16)
        load_btn.setFont(load_btn_font)

        layout.addWidget(title)
        layout.addWidget(self.tree_canvas)
        layout.addWidget(load_btn, alignment=Qt.AlignCenter)

        self.content_stack.addWidget(page)

    def apply_styles(self):
        """Apply professional styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            #navBar {
                background-color: #2c3e50;
                border-right: 1px solid #34495e;
            }
            #navBar QLabel {
                color: #ecf0f1;
            }
            #navButton {
                background-color: transparent;
                color: #bdc3c7;
                text-align: left;
                padding: 12px 20px;
                border: none;
                border-radius: 0;
                font-size: 16px;
            }
            #navButton:hover {
                background-color: #34495e;
                color: #ecf0f1;
            }
            #navButton:checked {
                background-color: #3498db;
                color: #ffffff;
                font-weight: bold;
            }
            #contentFrame {
                background-color: #ffffff;
            }
            QLabel {
                color: #2c3e50;
            }
            #actionButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 16px;
                min-width: 150px;
                min-height: 40px;
            }
            #actionButton:hover {
                background-color: #2980b9;
            }
            #actionButton:pressed {
                background-color: #1a6ea8;
            }
            QScrollArea {
                border: none;
            }
            QComboBox {
                min-width: 120px;
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
        """)

    def import_data(self, data_type):
        """Import data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {data_type} data file", "",
            "Data files (*.csv *.txt);;All files (*)"
        )

        if file_path:
            try:
                try:
                    df = pd.read_csv(file_path, header=None)
                    data = df.values.flatten()
                except:
                    data = np.loadtxt(file_path, delimiter=',')

                if data_type == 'normal':
                    self.normal_sample = data
                    self.data_status_label.setText(
                        f"Normal data loaded: {os.path.basename(file_path)}\nSamples: {len(self.normal_sample)}")
                else:
                    self.fault_sample = data
                    self.data_status_label.setText(
                        f"Fault data loaded: {os.path.basename(file_path)}\nSamples: {len(self.fault_sample)}")

                QMessageBox.information(self, "Success", f"{data_type} data imported successfully")

                for i in range(1, self.content_stack.count()):
                    self.nav_buttons[i].setEnabled(True)

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Data import failed: {str(e)}")
                self.data_status_label.setText(f"{data_type} data import failed")

    def show_time_domain_analysis(self):
        """Show time domain analysis"""
        if self.normal_sample is None or self.fault_sample is None:
            QMessageBox.warning(self, "Warning", "Please import both normal and fault data first")
            return

        if self.current_model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded, please load model first")
            return

        try:
            self.time_figure.clear()

            ax1 = self.time_figure.add_subplot(211)
            ax2 = self.time_figure.add_subplot(212)

            ax1.plot(self.normal_sample, color='#3498db', linewidth=1.2)
            ax1.set_title('Normal State Time Domain Signal', fontsize=16)
            ax1.set_xlabel('Sample Points', fontsize=14)
            ax1.set_ylabel('Amplitude', fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.plot(self.fault_sample, color='#e74c3c', linewidth=1.2)
            ax2.set_title('Fault State Time Domain Signal', fontsize=16)
            ax2.set_xlabel('Sample Points', fontsize=14)
            ax2.set_ylabel('Amplitude', fontsize=14)
            ax2.grid(True, linestyle='--', alpha=0.5)

            self.time_figure.tight_layout()
            self.time_canvas.draw()

            sample_data = self.fault_sample.reshape(1, -1)
            prediction = self.current_model.predict(sample_data)[0]
            fault_types = ["speed1", "speed2", "speed3", "unbalance fault1", "unbalance fault2"]
            if self.class_labels:
                predicted_class = fault_types[int(self.class_labels[prediction])]
            else:
                if 0 <= prediction < len(fault_types):
                    predicted_class = fault_types[prediction]
                else:
                    predicted_class = f"Class {prediction + 1}"

            self.time_prediction_label.setText(f"Prediction: {predicted_class}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Time domain analysis failed: {str(e)}")

    def calculate_time_domain_features(self, signal):
        """Calculate time domain features"""
        features = {}
        features['Peak'] = np.max(np.abs(signal))
        features['Mean'] = np.mean(signal)
        features['RMS'] = np.sqrt(np.mean(signal ** 2))
        features['Kurtosis'] = kurtosis(signal, fisher=False)
        features['Waveform Factor'] = features['RMS'] / np.mean(np.abs(signal))
        features['Impulse Factor'] = features['Peak'] / np.mean(np.abs(signal))
        features['Peak Factor'] = features['Peak'] / features['RMS']
        features['Kurtosis Factor'] = features['Kurtosis'] / (features['RMS'] ** 4)
        return features

    def show_features_analysis(self):
        """Show feature indicators analysis"""
        if self.normal_sample is None or self.fault_sample is None:
            QMessageBox.warning(self, "Warning", "Please import both normal and fault data first")
            return

        if self.current_model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded, please load model first")
            return

        try:
            normal_features = self.calculate_time_domain_features(self.normal_sample)
            fault_features = self.calculate_time_domain_features(self.fault_sample)

            self.features_figure.clear()
            ax = self.features_figure.add_subplot(111)

            feature_names = list(normal_features.keys())
            normal_values = list(normal_features.values())
            fault_values = list(fault_features.values())

            x = np.arange(len(feature_names))
            width = 0.35

            bars1 = ax.bar(x - width / 2, normal_values, width, label='Normal State', color='#3498db', alpha=0.7)
            bars2 = ax.bar(x + width / 2, fault_values, width, label='Fault State', color='#e74c3c', alpha=0.7)

            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=12)

            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=12)

            ax.set_title('Time Domain Feature Comparison', fontsize=16)
            ax.set_ylabel('Value', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            self.features_figure.tight_layout()
            self.features_canvas.draw()

            sample_data = self.fault_sample.reshape(1, -1)
            prediction = self.current_model.predict(sample_data)[0]
            fault_types = ["speed1", "speed2", "speed3", "unbalance fault1", "unbalance fault2"]
            if self.class_labels:
                predicted_class = fault_types[int(self.class_labels[prediction])]
            else:
                if 0 <= prediction < len(fault_types):
                    predicted_class = fault_types[prediction]
                else:
                    predicted_class = f"Class {prediction + 1}"

            self.features_prediction_label.setText(f"Prediction: {predicted_class}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Feature analysis failed: {str(e)}")

    def show_spectral_kurtosis(self):
        """Show spectral kurtosis analysis"""
        if self.normal_sample is None or self.fault_sample is None:
            QMessageBox.warning(self, "Warning", "Please import both normal and fault data first")
            return

        if self.current_model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded, please load model first")
            return

        try:
            f_n, t_n, Sxx_n = spectrogram(self.normal_sample, fs=self.sampling_rate, nperseg=256)
            kurt_n = kurtosis(np.abs(Sxx_n), axis=1)

            f_f, t_f, Sxx_f = spectrogram(self.fault_sample, fs=self.sampling_rate, nperseg=256)
            kurt_f = kurtosis(np.abs(Sxx_f), axis=1)

            self.kurtosis_figure.clear()
            ax = self.kurtosis_figure.add_subplot(111)

            ax.plot(f_n, kurt_n, color='#3498db', linewidth=1.5, label='Normal State')
            ax.plot(f_f, kurt_f, color='#e74c3c', linewidth=1.5, label='Fault State')

            ax.set_title('Spectral Kurtosis Comparison', fontsize=16)
            ax.set_xlabel('Frequency (Hz)', fontsize=14)
            ax.set_ylabel('Kurtosis Value', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

            self.kurtosis_figure.tight_layout()
            self.kurtosis_canvas.draw()

            sample_data = self.fault_sample.reshape(1, -1)
            prediction = self.current_model.predict(sample_data)[0]
            fault_types = ["speed1", "speed2", "speed3", "unbalance fault1", "unbalance fault2"]
            if self.class_labels:
                predicted_class = fault_types[int(self.class_labels[prediction])]
            else:
                if 0 <= prediction < len(fault_types):
                    predicted_class = fault_types[prediction]
                else:
                    predicted_class = f"Class {prediction + 1}"

            self.kurtosis_prediction_label.setText(f"Prediction: {predicted_class}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Spectral kurtosis analysis failed: {str(e)}")

    def show_confusion_matrix(self):
        """Display confusion matrix"""
        self.confusion_figure.clear()
        ax = self.confusion_figure.add_subplot(111)

        algorithm = "cart" if self.algorithm_combo.currentIndex() == 0 else "id3"
        cm_path = f'result/confusion_matrix_{algorithm}_normalized.png'

        if os.path.exists(cm_path):
            try:
                cm_img = plt.imread(cm_path)
                ax.imshow(cm_img)
                ax.set_title(f'Normalized Confusion Matrix ({self.algorithm_combo.currentText()})', fontsize=16)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'Loading failed: {str(e)}', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Confusion matrix image not found for {self.algorithm_combo.currentText()}',
                    ha='center', va='center')
            ax.axis('off')

        self.confusion_canvas.draw()

    def show_learning_curve(self):
        """Display learning curve"""
        self.learning_curve_figure.clear()
        ax = self.learning_curve_figure.add_subplot(111)

        algorithm = "cart" if self.algorithm_combo.currentIndex() == 0 else "id3"
        lc_path = f'result/learning_curve_{algorithm}.png'

        if os.path.exists(lc_path):
            try:
                lc_img = plt.imread(lc_path)
                ax.imshow(lc_img)
                ax.set_title(f'Learning Curve ({self.algorithm_combo.currentText()})', fontsize=16)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'Loading failed: {str(e)}', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Learning curve image not found for {self.algorithm_combo.currentText()}',
                    ha='center', va='center')
            ax.axis('off')

        self.learning_curve_canvas.draw()

    def show_accuracy(self):
        """Display accuracy"""
        if self.algorithm_combo.currentIndex() == 0:
            accuracy = "0.9385"
            algorithm = "CART"
        else:
            accuracy = "0.9511"
            algorithm = "ID3"

        self.accuracy_label.setText(f"{algorithm} algorithm accuracy: {accuracy}")

    def show_decision_tree(self):
        """Display decision tree visualization"""
        if self.current_model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded")
            return

        self.tree_figure.clear()
        ax = self.tree_figure.add_subplot(111)

        try:
            plot_tree(
                self.current_model,
                filled=True,
                rounded=True,
                feature_names=[f"Feature_{i}" for i in range(self.current_model.n_features_in_)],
                class_names=self.class_labels if self.class_labels else [f"Class_{i}" for i in
                                                                         range(self.current_model.n_classes_)],
                ax=ax,
                fontsize=10
            )
            ax.set_title(f'Decision Tree Structure ({self.algorithm_combo.currentText()})', fontsize=16)
            self.tree_figure.tight_layout()
        except Exception as e:
            ax.text(0.5, 0.5, f"Decision tree visualization failed: {str(e)}", ha='center', va='center')
            ax.axis('off')

        self.tree_canvas.draw()

    def predict_fault(self):
        """Perform fault prediction"""
        if self.current_model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded, please load model first")
            return

        if self.fault_sample is None:
            QMessageBox.warning(self, "Warning", "Please import fault data first")
            return

        sample_data = self.fault_sample.reshape(1, -1)

        prediction = self.current_model.predict(sample_data)[0]
        proba = self.current_model.predict_proba(sample_data)[0]

        fault_types = ["speed1", "speed2", "speed3", "unbalance fault1", "unbalance fault2"]
        if self.class_labels:
            predicted_class = fault_types[int(self.class_labels[prediction])]
        else:
            if 0 <= prediction < len(fault_types):
                predicted_class = fault_types[prediction]
            else:
                predicted_class = f"Class {prediction + 1}"

        result_text = f"Prediction: {predicted_class}"
        QMessageBox.information(self, "Prediction Result", result_text)


def main():
    app = QApplication(sys.argv)
    window = GearHealthMonitoringSystem()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
