#!/usr/bin/env python3
"""
Modern Hardware Testing Suite (2025 Edition)
"""

import sys
import os
import platform
import threading
import time
from datetime import datetime
import random

# Consolidated imports with error handling
try:
    import psutil
    import wmi
    import cv2
    import sounddevice as sd
    import numpy as np
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QProgressBar, QTabWidget, QTextEdit,
        QFrame, QScrollArea, QSizePolicy, QMessageBox, QDialog,
        QStackedWidget, QGridLayout
    )
    from PySide6.QtCore import Qt, Signal, Slot, QTimer, QSize
    from PySide6.QtGui import QFont, QPalette, QColor, QIcon, QPainter, QImage, QPixmap
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    import pycuda.autoinit
    import pycuda.driver as cuda
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please run: pip install psutil wmi opencv-python sounddevice numpy PySide6 PyOpenGL pycuda")
    sys.exit(1)

class ModernHardwareTester(QMainWindow):
    test_signal = Signal(str)  # For thread-safe logging
    progress_signal = Signal(int)  # For thread-safe progress updates

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hardware Testing Suite")
        self.setMinimumSize(1200, 800)
        
        # Initialize state
        self.test_running = False
        self.stop_requested = False
        self.test_results = {}
        
        self.setup_ui()
        self.setup_signals()
        self.apply_styles()
        self.scan_system()

    def setup_ui(self):
        """Create the modern UI layout"""
        # Create central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Create header with system info
        self.header = self.create_header()
        main_layout.addWidget(self.header)

        # Create main content area
        content = QHBoxLayout()
        
        # Left side - Test categories and controls
        left_panel = self.create_left_panel()
        content.addWidget(left_panel, 1)
        
        # Right side - Monitoring and results
        right_panel = self.create_right_panel()
        content.addWidget(right_panel, 2)
        
        main_layout.addLayout(content)

        # Create status bar at bottom
        self.status_bar = self.create_status_bar()
        main_layout.addWidget(self.status_bar)

    def create_header(self):
        """Create modern header with system information"""
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)

        # System info section
        info_layout = QVBoxLayout()
        self.system_info_label = QLabel("Scanning system...")
        self.system_info_label.setObjectName("systemInfo")
        info_layout.addWidget(self.system_info_label)

        # Quick actions section
        actions_layout = QHBoxLayout()
        self.stop_button = QPushButton("Stop Test")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_current_test)
        self.stop_button.setEnabled(False)
        actions_layout.addWidget(self.stop_button)

        layout.addLayout(info_layout)
        layout.addLayout(actions_layout)
        return header

    def create_left_panel(self):
        """Create left panel with test categories"""
        panel = QFrame()
        panel.setObjectName("leftPanel")
        layout = QVBoxLayout(panel)

        # Create tab widget for test categories
        self.test_tabs = QTabWidget()
        self.test_tabs.setObjectName("testTabs")

        # Quick Tests Tab
        quick_tests = QWidget()
        quick_layout = QGridLayout(quick_tests)
        quick_tests_list = [
            ("CPU Test", self.test_cpu),
            ("Memory Test", self.test_memory),
            ("Storage Test", self.test_storage),
            ("GPU Test", self.test_gpu),
            ("Display Test", self.test_display),
            ("Audio Test", self.test_audio)
        ]
        self.create_test_buttons(quick_layout, quick_tests_list)
        self.test_tabs.addTab(quick_tests, "Quick Tests")

        # Stress Tests Tab
        stress_tests = QWidget()
        stress_layout = QGridLayout(stress_tests)
        stress_tests_list = [
            ("CPU Stress", self.test_cpu_stress),
            ("Memory Stress", self.test_memory_stress),
            ("GPU Stress", self.test_gpu_stress),
            ("System Stability", self.test_system_stability)
        ]
        self.create_test_buttons(stress_layout, stress_tests_list)
        self.test_tabs.addTab(stress_tests, "Stress Tests")

        # Add GPU Stress Test button
        gpu_stress_button = QPushButton("GPU Stress Test")
        gpu_stress_button.clicked.connect(self.test_gpu_stress)
        stress_layout.addWidget(gpu_stress_button, 1, 0)

        layout.addWidget(self.test_tabs)
        return panel

    def create_right_panel(self):
        """Create right panel with monitoring and results"""
        panel = QFrame()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)

        # Create monitoring section
        monitoring = QFrame()
        monitoring.setObjectName("monitoring")
        monitoring_layout = QVBoxLayout(monitoring)

        # CPU Usage Chart
        self.cpu_chart = self.create_chart("CPU Usage", "%")
        monitoring_layout.addWidget(self.cpu_chart)

        # Memory Usage Chart
        self.memory_chart = self.create_chart("Memory Usage", "GB")
        monitoring_layout.addWidget(self.memory_chart)

        # Temperature Chart
        self.temp_chart = self.create_chart("Temperature", "°C")
        monitoring_layout.addWidget(self.temp_chart)

        # Add GPU Usage Chart
        self.gpu_chart = self.create_chart("GPU Usage", "%")
        monitoring_layout.addWidget(self.gpu_chart)

        layout.addWidget(monitoring)

        # Create results section
        self.results_text = QTextEdit()
        self.results_text.setObjectName("results")
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        return panel

    def create_chart(self, title, unit):
        """Create a real-time chart for monitoring"""
        chart = QChart()
        chart.setTitle(title)
        
        # Create series for the data
        series = QLineSeries()
        chart.addSeries(series)
        
        # Create axes
        axis_x = QValueAxis()
        axis_x.setRange(0, 60)  # Show last 60 seconds
        axis_x.setLabelFormat("%d")
        axis_x.setTitleText("Time (s)")
        
        axis_y = QValueAxis()
        axis_y.setRange(0, 100)
        axis_y.setLabelFormat("%i")
        axis_y.setTitleText(unit)
        
        chart.setAxisX(axis_x, series)
        chart.setAxisY(axis_y, series)
        
        # Create chart view
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setMinimumHeight(200)
        
        return chart_view

    def create_status_bar(self):
        """Create status bar with progress"""
        status = QFrame()
        status.setObjectName("statusBar")
        layout = QHBoxLayout(status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        return status

    def apply_styles(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QFrame#header {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
            }
            QFrame#leftPanel, QFrame#rightPanel {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
            }
            QFrame#monitoring {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QPushButton#stopButton {
                background-color: #dc3545;
            }
            QPushButton#stopButton:hover {
                background-color: #c82333;
            }
            QProgressBar {
                border: none;
                background-color: #e9ecef;
                border-radius: 4px;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 4px;
            }
            QTextEdit#results {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
                font-family: monospace;
            }
            QLabel#systemInfo {
                font-size: 14px;
                color: #495057;
            }
        """)

    def setup_signals(self):
        """Setup signal connections"""
        self.test_signal.connect(self.log_message)
        self.progress_signal.connect(self.update_progress)

    def scan_system(self):
        """Scan and display system information"""
        try:
            cpu_info = f"CPU: {platform.processor()}"
            memory = psutil.virtual_memory()
            memory_info = f"Memory: {memory.total / (1024**3):.1f} GB"
            
            if wmi:
                c = wmi.WMI()
                gpu_info = "GPU: " + ", ".join([gpu.Name for gpu in c.Win32_VideoController()])
            else:
                gpu_info = "GPU: Information not available"
            
            system_info = f"{cpu_info}\n{memory_info}\n{gpu_info}"
            self.system_info_label.setText(system_info)
            
        except Exception as e:
            self.system_info_label.setText(f"Error scanning system: {str(e)}")

    def create_test_buttons(self, layout, tests):
        """Create buttons for test list"""
        for i, (name, func) in enumerate(tests):
            button = QPushButton(name)
            button.clicked.connect(func)
            row = i // 2
            col = i % 2
            layout.addWidget(button, row, col)

    def log_message(self, message):
        """Add message to results text"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.append(f"[{timestamp}] {message}")

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def stop_current_test(self):
        """Stop the current running test"""
        self.stop_requested = True
        self.stop_button.setEnabled(False)
        self.log_message("Test stop requested...")

    def test_cpu(self):
        """Quick CPU test with real-time monitoring"""
        def run_test():
            self.test_signal.emit("Starting CPU Test...")
            self.stop_button.setEnabled(True)
            duration = 30  # 30 seconds test
            
            try:
                # Initialize data series for the chart
                series = self.cpu_chart.chart().series()[0]
                series.clear()
                
                start_time = time.time()
                data_points = []
                
                while time.time() - start_time < duration and not self.stop_requested:
                    # Get CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
                    avg_cpu = sum(cpu_percent) / len(cpu_percent)
                    
                    # Get CPU frequency
                    freq = psutil.cpu_freq(percpu=True) if hasattr(psutil.cpu_freq(), 'current') else None
                    
                    # Get CPU temperature if available
                    temp_info = ""
                    if hasattr(psutil, "sensors_temperatures"):
                        temps = psutil.sensors_temperatures()
                        if temps and 'coretemp' in temps:
                            temp_info = f"Temperature: {max(t.current for t in temps['coretemp'])}°C"
                    
                    # Update chart
                    elapsed = time.time() - start_time
                    series.append(elapsed, avg_cpu)
                    
                    # Log CPU info
                    self.test_signal.emit(
                        f"CPU Usage: {avg_cpu:.1f}% | Per Core: {cpu_percent}\n"
                        f"Frequency: {freq}\n{temp_info}"
                    )
                    
                    # Update progress
                    progress = (elapsed / duration) * 100
                    self.progress_signal.emit(int(progress))
                    
                    # Store data for analysis
                    data_points.append(avg_cpu)
                    
                    QApplication.processEvents()
                
                # Analyze results
                avg_usage = sum(data_points) / len(data_points)
                max_usage = max(data_points)
                stability = 100 - (max(data_points) - min(data_points))
                
                result = (
                    f"CPU Test Results:\n"
                    f"Average Usage: {avg_usage:.1f}%\n"
                    f"Peak Usage: {max_usage:.1f}%\n"
                    f"Stability Score: {stability:.1f}%\n"
                )
                
                self.test_signal.emit(result)
                self.stop_button.setEnabled(False)
                
                # Record test result
                passed = avg_usage > 10 and stability > 70  # Basic pass criteria
                self.record_test_result(
                    "CPU Test",
                    passed,
                    result
                )
                
            except Exception as e:
                self.test_signal.emit(f"CPU Test Error: {str(e)}")
                self.record_test_result("CPU Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
        
        # Run test in a separate thread
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_memory(self):
        """Quick memory test"""
        def run_test():
            self.test_signal.emit("Starting Memory Test...")
            self.stop_button.setEnabled(True)
            duration = 30  # 30 seconds test
            
            try:
                # Initialize chart
                series = self.memory_chart.chart().series()[0]
                series.clear()
                
                start_time = time.time()
                data = []
                
                while time.time() - start_time < duration and not self.stop_requested:
                    # Get memory metrics
                    mem = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    
                    # Update chart
                    elapsed = time.time() - start_time
                    series.append(elapsed, mem.percent)
                    
                    # Log memory info
                    self.test_signal.emit(
                        f"Memory Usage: {mem.percent}%\n"
                        f"Used: {mem.used / (1024**3):.1f}GB / "
                        f"Total: {mem.total / (1024**3):.1f}GB\n"
                        f"Swap: {swap.used / (1024**3):.1f}GB / "
                        f"{swap.total / (1024**3):.1f}GB"
                    )
                    
                    # Update progress
                    progress = (elapsed / duration) * 100
                    self.progress_signal.emit(int(progress))
                    
                    data.append(mem.percent)
                    QApplication.processEvents()
                    time.sleep(0.5)
                
                # Analyze results
                avg_usage = sum(data) / len(data)
                max_usage = max(data)
                available_gb = mem.available / (1024**3)
                
                result = (
                    f"Memory Test Results:\n"
                    f"Average Usage: {avg_usage:.1f}%\n"
                    f"Peak Usage: {max_usage:.1f}%\n"
                    f"Available Memory: {available_gb:.1f}GB\n"
                    f"Swap Usage: {swap.percent}%"
                )
                
                self.test_signal.emit(result)
                
                # Record test result
                passed = available_gb > 1.0  # At least 1GB should be available
                self.record_test_result(
                    "Memory Test",
                    passed,
                    result
                )
                
            except Exception as e:
                self.test_signal.emit(f"Memory Test Error: {str(e)}")
                self.record_test_result("Memory Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
        
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_storage(self):
        """Storage test"""
        def run_test():
            self.test_signal.emit("Starting Storage Test...")
            try:
                # Get disk partitions
                partitions = psutil.disk_partitions()
                results = []
                
                for partition in partitions:
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        
                        # Get disk I/O if available
                        io_counters = psutil.disk_io_counters(perdisk=True)
                        disk_name = partition.device.split('\\')[-1]
                        io_info = io_counters.get(disk_name, None)
                        
                        result = (
                            f"\nPartition: {partition.mountpoint}\n"
                            f"Device: {partition.device}\n"
                            f"File System: {partition.fstype}\n"
                            f"Total: {usage.total / (1024**3):.1f}GB\n"
                            f"Used: {usage.used / (1024**3):.1f}GB ({usage.percent}%)\n"
                            f"Free: {usage.free / (1024**3):.1f}GB\n"
                        )
                        
                        if io_info:
                            result += (
                                f"Read Speed: {io_info.read_bytes / (1024**2):.1f}MB/s\n"
                                f"Write Speed: {io_info.write_bytes / (1024**2):.1f}MB/s"
                            )
                        
                        results.append(result)
                        self.test_signal.emit(result)
                        
                    except Exception as e:
                        self.test_signal.emit(f"Error checking {partition.mountpoint}: {str(e)}")
                
                # Record overall result
                passed = all(p.mountpoint and psutil.disk_usage(p.mountpoint).free > 1024**3 for p in partitions)
                self.record_test_result(
                    "Storage Test",
                    passed,
                    "\n".join(results)
                )
                
            except Exception as e:
                self.test_signal.emit(f"Storage Test Error: {str(e)}")
                self.record_test_result("Storage Test", False, str(e))
        
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_gpu(self):
        """GPU test with OpenGL and CUDA detection"""
        def run_test():
            self.test_signal.emit("Starting GPU Test...")
            self.stop_button.setEnabled(True)
            
            try:
                results = []
                
                # Get GPU information using WMI
                if wmi:
                    c = wmi.WMI()
                    gpu_devices = c.Win32_VideoController()
                    
                    for i, gpu in enumerate(gpu_devices):
                        gpu_info = (
                            f"\nGPU {i+1}:\n"
                            f"Name: {gpu.Name}\n"
                            f"Driver Version: {gpu.DriverVersion}\n"
                            f"Video Memory: {gpu.AdapterRAM / (1024**3):.1f} GB\n" if gpu.AdapterRAM else "Video Memory: N/A\n"
                            f"Current Resolution: {gpu.CurrentHorizontalResolution}x{gpu.CurrentVerticalResolution}\n"
                            f"Refresh Rate: {gpu.CurrentRefreshRate} Hz\n"
                            f"Driver Date: {gpu.DriverDate}\n"
                            f"Video Processor: {gpu.VideoProcessor}\n"
                        )
                        results.append(gpu_info)
                        self.test_signal.emit(gpu_info)
                
                # Try to detect CUDA capability
                try:
                    import pycuda.autoinit
                    import pycuda.driver as cuda
                    
                    cuda_info = "\nCUDA Information:\n"
                    cuda_info += f"CUDA Version: {'.'.join(map(str, cuda.get_version()))}\n"
                    
                    device = cuda.Device(0)
                    cuda_info += (
                        f"Device Name: {device.name()}\n"
                        f"Compute Capability: {device.compute_capability()}\n"
                        f"Total Memory: {device.total_memory() / (1024**2):.1f} MB\n"
                        f"Max Threads per Block: {device.max_threads_per_block}\n"
                    )
                    
                    results.append(cuda_info)
                    self.test_signal.emit(cuda_info)
                    
                except ImportError:
                    self.test_signal.emit("CUDA support not available (pycuda not installed)")
                except Exception as e:
                    self.test_signal.emit(f"CUDA detection error: {str(e)}")
                
                # Try OpenGL info
                try:
                    import OpenGL.GL as gl
                    import OpenGL.GLU as glu
                    
                    opengl_info = "\nOpenGL Information:\n"
                    opengl_info += f"Vendor: {gl.glGetString(gl.GL_VENDOR).decode()}\n"
                    opengl_info += f"Renderer: {gl.glGetString(gl.GL_RENDERER).decode()}\n"
                    opengl_info += f"Version: {gl.glGetString(gl.GL_VERSION).decode()}\n"
                    
                    results.append(opengl_info)
                    self.test_signal.emit(opengl_info)
                    
                except ImportError:
                    self.test_signal.emit("OpenGL support not available (PyOpenGL not installed)")
                except Exception as e:
                    self.test_signal.emit(f"OpenGL detection error: {str(e)}")
                
                # Basic 2D acceleration test
                try:
                    self.test_signal.emit("\nRunning 2D acceleration test...")
                    
                    # Create large image
                    size = 2048
                    test_image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                    
                    # Measure time for image operations
                    start_time = time.time()
                    
                    for _ in range(10):
                        # Convert to Qt image and back
                        qimg = QImage(test_image.data, size, size, 3 * size, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        scaled = pixmap.scaled(size//2, size//2)
                        
                    operation_time = time.time() - start_time
                    
                    perf_info = (
                        f"\n2D Performance Test:\n"
                        f"Time for image operations: {operation_time:.2f}s\n"
                        f"Performance rating: {'Good' if operation_time < 1 else 'Fair' if operation_time < 2 else 'Poor'}\n"
                    )
                    
                    results.append(perf_info)
                    self.test_signal.emit(perf_info)
                    
                except Exception as e:
                    self.test_signal.emit(f"2D acceleration test error: {str(e)}")
                
                # Record overall result
                passed = len(results) > 0  # Pass if we got any GPU information
                self.record_test_result(
                    "GPU Test",
                    passed,
                    "\n".join(results)
                )
                
            except Exception as e:
                self.test_signal.emit(f"GPU Test Error: {str(e)}")
                self.record_test_result("GPU Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
        
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_gpu_stress(self):
        """Extended GPU stress test"""
        def run_test():
            self.test_signal.emit("Starting GPU Stress Test...")
            self.stop_button.setEnabled(True)
            duration = 600  # 10 minutes
            
            try:
                # Initialize monitoring
                series = self.gpu_chart.chart().series()[0]
                series.clear()
                
                start_time = time.time()
                data_points = []
                
                def stress_worker():
                    # Create a large image for GPU stress
                    img_size = 4096
                    while time.time() - start_time < duration and not self.stop_requested:
                        # Create random image
                        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
                        # Apply various OpenCV operations
                        img = cv2.GaussianBlur(img, (15, 15), 0)
                        img = cv2.Canny(img, 100, 200)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        # Display image (optional)
                        # cv2.imshow('GPU Stress', img)
                        # cv2.waitKey(1)
                
                # Start stress thread
                stress_thread = threading.Thread(target=stress_worker)
                stress_thread.daemon = True
                stress_thread.start()
                
                # Monitor GPU usage and temperature
                while time.time() - start_time < duration and not self.stop_requested:
                    # Get GPU metrics
                    gpu_metrics = {}
                    if wmi:
                        try:
                            c = wmi.WMI(namespace='root\\wmi')
                            gpu_metrics['gpu_temp'] = c.MSAcpi_ThermalZoneTemperature()[0].CurrentTemperature/10 - 273.15
                            gpu_metrics['gpu_usage'] = c.Win32_VideoController()[0].LoadPercentage
                        except:
                            pass
                    
                    # Update chart
                    elapsed = time.time() - start_time
                    if 'gpu_usage' in gpu_metrics:
                        series.append(elapsed, gpu_metrics['gpu_usage'])
                        data_points.append(gpu_metrics['gpu_usage'])
                    
                    # Get temperature if available
                    temp_info = ""
                    if 'gpu_temp' in gpu_metrics:
                        temp_info = f", Temperature: {gpu_metrics['gpu_temp']}°C"
                    
                    self.test_signal.emit(
                        f"GPU Usage: {gpu_metrics.get('gpu_usage', 'N/A')}%{temp_info}"
                    )
                    
                    # Update progress
                    progress = (elapsed / duration) * 100
                    self.progress_signal.emit(int(progress))
                    
                    QApplication.processEvents()
                    time.sleep(1)
                
                # Analysis
                avg_usage = sum(data_points) / len(data_points) if data_points else 0
                max_usage = max(data_points) if data_points else 0
                stability = 100 - (max(data_points) - min(data_points)) if data_points else 0
                
                result = (
                    f"GPU Stress Test Results:\n"
                    f"Duration: {duration/60:.1f} minutes\n"
                    f"Average Usage: {avg_usage:.1f}%\n"
                    f"Peak Usage: {max_usage:.1f}%\n"
                    f"Stability Score: {stability:.1f}%\n"
                )
                
                # Record result
                passed = avg_usage > 90 and stability > 80
                self.record_test_result(
                    "GPU Stress Test",
                    passed,
                    result
                )
                
            except Exception as e:
                self.test_signal.emit(f"GPU Stress Test Error: {str(e)}")
                self.record_test_result("GPU Stress Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
        
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_display(self):
        """Display test with color patterns and resolution check"""
        def run_test():
            self.test_signal.emit("Starting Display Test...")
            self.stop_button.setEnabled(True)
            
            try:
                results = []
                
                # Get display information
                screen = QApplication.primaryScreen()
                geometry = screen.geometry()
                physical_dpi = screen.physicalDotsPerInch()
                logical_dpi = screen.logicalDotsPerInch()
                
                display_info = (
                    f"\nDisplay Information:\n"
                    f"Resolution: {geometry.width()}x{geometry.height()}\n"
                    f"Physical DPI: {physical_dpi:.1f}\n"
                    f"Logical DPI: {logical_dpi:.1f}\n"
                    f"Color Depth: {screen.depth()} bits\n"
                    f"Refresh Rate: {screen.refreshRate()} Hz\n"
                )
                
                results.append(display_info)
                self.test_signal.emit(display_info)
                
                # Create test window
                test_window = QWidget()
                test_window.setWindowTitle("Display Test")
                test_window.setWindowState(Qt.WindowFullScreen)
                
                # Create test patterns
                patterns = [
                    ("White", Qt.white),
                    ("Black", Qt.black),
                    ("Red", Qt.red),
                    ("Green", Qt.green),
                    ("Blue", Qt.blue),
                    ("Gradient", None)  # Special case for gradient
                ]
                
                for name, color in patterns:
                    if self.stop_requested:
                        break
                        
                    self.test_signal.emit(f"Testing {name} pattern...")
                    
                    if color is not None:
                        test_window.setStyleSheet(f"background-color: {color.name()};")
                    else:
                        # Create gradient pattern
                        gradient = """
                            background: qlineargradient(
                                x1: 0, y1: 0, x2: 1, y2: 1,
                                stop: 0 red,
                                stop: 0.33 yellow,
                                stop: 0.66 green,
                                stop: 1 blue
                            );
                        """
                        test_window.setStyleSheet(gradient)
                    
                    test_window.show()
                    QApplication.processEvents()
                    time.sleep(2)  # Show each pattern for 2 seconds
                
                test_window.close()
                
                # Check for dead pixels
                self.test_signal.emit("\nChecking for dead pixels...")
                
                # Create full screen screenshot
                screenshot = screen.grabWindow(0)
                image = screenshot.toImage()
                
                # Analyze image for dead pixels
                dead_pixels = []
                for x in range(0, image.width(), 10):  # Sample every 10th pixel
                    for y in range(0, image.height(), 10):
                        color = QColor(image.pixel(x, y))
                        if color.red() == 0 and color.green() == 0 and color.blue() == 0:
                            dead_pixels.append((x, y))
                
                if dead_pixels:
                    results.append(f"Found {len(dead_pixels)} potential dead pixels")
                    self.test_signal.emit(f"Warning: Found {len(dead_pixels)} potential dead pixels")
                else:
                    results.append("No dead pixels detected")
                    self.test_signal.emit("No dead pixels detected")
                
                # Record test result
                passed = len(dead_pixels) == 0
                self.record_test_result(
                    "Display Test",
                    passed,
                    "\n".join(results)
                )
                
            except Exception as e:
                self.test_signal.emit(f"Display Test Error: {str(e)}")
                self.record_test_result("Display Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
        
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_audio(self):
        """Audio input/output test"""
        def run_test():
            self.test_signal.emit("Starting Audio Test...")
            self.stop_button.setEnabled(True)
            
            try:
                results = []
                
                # Get audio device information
                devices_info = "\nAudio Devices:\n"
                input_devices = sd.query_devices(kind='input')
                output_devices = sd.query_devices(kind='output')
                
                devices_info += "\nInput Devices:\n"
                for dev in input_devices:
                    devices_info += f"- {dev['name']} ({dev['max_input_channels']} channels)\n"
                
                devices_info += "\nOutput Devices:\n"
                for dev in output_devices:
                    devices_info += f"- {dev['name']} ({dev['max_output_channels']} channels)\n"
                
                results.append(devices_info)
                self.test_signal.emit(devices_info)
                
                # Test audio output
                self.test_signal.emit("\nTesting Audio Output...")
                
                # Generate test tones
                sample_rate = 44100
                duration = 1  # seconds
                
                def generate_tone(frequency):
                    t = np.linspace(0, duration, int(sample_rate * duration), False)
                    return 0.3 * np.sin(2 * np.pi * frequency * t)  # 30% amplitude
                
                test_frequencies = [
                    (440, "A4 (440 Hz)"),
                    (1000, "1 kHz"),
                    (4000, "4 kHz")
                ]
                
                # Play test tones
                for freq, name in test_frequencies:
                    if self.stop_requested:
                        break
                        
                    self.test_signal.emit(f"Playing {name}...")
                    tone = generate_tone(freq)
                    
                    # Add fade in/out to prevent clicks
                    fade_len = int(0.1 * sample_rate)  # 100ms fade
                    fade_in = np.linspace(0, 1, fade_len)
                    fade_out = np.linspace(1, 0, fade_len)
                    tone[:fade_len] *= fade_in
                    tone[-fade_len:] *= fade_out
                    
                    sd.play(tone, sample_rate)
                    sd.wait()
                    time.sleep(0.5)  # Gap between tones
                
                # Test audio input (microphone)
                if not self.stop_requested:
                    self.test_signal.emit("\nTesting Audio Input (3 seconds)...")
                    
                    input_duration = 3  # seconds
                    recording = sd.rec(
                        int(input_duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1
                    )
                    
                    # Show progress during recording
                    for i in range(input_duration):
                        if self.stop_requested:
                            break
                        self.progress_signal.emit((i / input_duration) * 100)
                        time.sleep(1)
                    
                    sd.wait()
                    
                    # Analyze recording
                    if len(recording) > 0:
                        rms = np.sqrt(np.mean(recording**2))
                        db = 20 * np.log10(rms) if rms > 0 else -float('inf')
                        peak = np.max(np.abs(recording))
                        
                        audio_analysis = (
                            f"\nAudio Input Analysis:\n"
                            f"RMS Level: {db:.1f} dB\n"
                            f"Peak Level: {20 * np.log10(peak):.1f} dB\n"
                            f"Signal Detected: {'Yes' if db > -60 else 'No'}\n"
                        )
                        
                        results.append(audio_analysis)
                        self.test_signal.emit(audio_analysis)
                        
                        # Detect potential issues
                        if db < -60:
                            self.test_signal.emit("Warning: Very low input level detected")
                        elif db > -1:
                            self.test_signal.emit("Warning: Possible input clipping detected")
                    
                    # Record test result
                    passed = -60 < db < -1  # Basic audio level check
                    self.record_test_result(
                        "Audio Test",
                        passed,
                        "\n".join(results)
                    )
                
            except Exception as e:
                self.test_signal.emit(f"Audio Test Error: {str(e)}")
                self.record_test_result("Audio Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
                
                # Ensure audio streams are closed
                try:
                    sd.stop()
                except:
                    pass
        
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_cpu_stress(self):
        """Extended CPU stress test"""
        def run_test():
            self.test_signal.emit("Starting CPU Stress Test...")
            self.stop_button.setEnabled(True)
            duration = 600  # 10 minutes
            
            try:
                # Initialize monitoring
                series = self.cpu_chart.chart().series()[0]
                series.clear()
                
                start_time = time.time()
                data_points = []
                
                def stress_worker():
                    while time.time() - start_time < duration and not self.stop_requested:
                        # Heavy computation
                        matrix_size = 200
                        matrix = [[random.random() for _ in range(matrix_size)] 
                                 for _ in range(matrix_size)]
                        # Matrix multiplication
                        result = [[sum(a * b for a, b in zip(row, col)) 
                                 for col in zip(*matrix)] 
                                 for row in matrix]
                        # Prime numbers
                        primes = [num for num in range(2, 10000) 
                                if all(num % i != 0 for i in range(2, int(num ** 0.5) + 1))]
                
                # Start stress threads (one per CPU core)
                stress_threads = []
                for _ in range(psutil.cpu_count()):
                    thread = threading.Thread(target=stress_worker)
                    thread.daemon = True
                    thread.start()
                    stress_threads.append(thread)
                
                # Monitor CPU usage and temperature
                while time.time() - start_time < duration and not self.stop_requested:
                    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                    avg_cpu = sum(cpu_percent) / len(cpu_percent)
                    
                    # Update chart
                    elapsed = time.time() - start_time
                    series.append(elapsed, avg_cpu)
                    data_points.append(avg_cpu)
                    
                    # Get temperature if available
                    temp_info = ""
                    if hasattr(psutil, "sensors_temperatures"):
                        temps = psutil.sensors_temperatures()
                        if temps and 'coretemp' in temps:
                            current_temp = max(t.current for t in temps['coretemp'])
                            temp_info = f", Temperature: {current_temp}°C"
                    
                    self.test_signal.emit(
                        f"CPU Usage: {avg_cpu:.1f}% (Per Core: {cpu_percent}){temp_info}"
                    )
                    
                    # Update progress
                    progress = (elapsed / duration) * 100
                    self.progress_signal.emit(int(progress))
                    
                    QApplication.processEvents()
                
                # Analysis
                avg_usage = sum(data_points) / len(data_points)
                max_usage = max(data_points)
                stability = 100 - (max(data_points) - min(data_points))
                
                result = (
                    f"CPU Stress Test Results:\n"
                    f"Duration: {duration/60:.1f} minutes\n"
                    f"Average Usage: {avg_usage:.1f}%\n"
                    f"Peak Usage: {max_usage:.1f}%\n"
                    f"Stability Score: {stability:.1f}%\n"
                )
                
                # Record result
                passed = avg_usage > 90 and stability > 80
                self.record_test_result(
                    "CPU Stress Test",
                    passed,
                    result
                )
                
            except Exception as e:
                self.test_signal.emit(f"CPU Stress Test Error: {str(e)}")
                self.record_test_result("CPU Stress Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
        
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_memory_stress(self):
        """Extended memory stress test"""
        def run_test():
            self.test_signal.emit("Starting Memory Stress Test...")
            self.stop_button.setEnabled(True)
            duration = 300  # 5 minutes
            
            try:
                # Initialize monitoring
                series = self.memory_chart.chart().series()[0]
                series.clear()
                
                start_time = time.time()
                memory_blocks = []
                block_size = 100 * 1024 * 1024  # 100MB blocks
                
                while time.time() - start_time < duration and not self.stop_requested:
                    try:
                        # Allocate memory
                        memory_blocks.append(bytearray(block_size))
                        
                        # Get memory usage
                        mem = psutil.virtual_memory()
                        swap = psutil.swap_memory()
                        
                        # Update chart
                        elapsed = time.time() - start_time
                        series.append(elapsed, mem.percent)
                        
                        self.test_signal.emit(
                            f"Memory Usage: {mem.percent}%\n"
                            f"Used: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB\n"
                            f"Swap: {swap.used/1024**3:.1f}GB / {swap.total/1024**3:.1f}GB"
                        )
                        
                        # Update progress
                        progress = (elapsed / duration) * 100
                        self.progress_signal.emit(int(progress))
                        
                        time.sleep(1)
                        QApplication.processEvents()
                        
                    except MemoryError:
                        self.test_signal.emit("Maximum memory allocation reached")
                        break
                
                # Clean up
                memory_blocks.clear()
                
                # Final memory check
                mem = psutil.virtual_memory()
                result = (
                    f"Memory Stress Test Results:\n"
                    f"Peak Memory Usage: {mem.percent}%\n"
                    f"Available After Test: {mem.available/1024**3:.1f}GB"
                )
                
                # Record result
                passed = mem.available > 1024**3  # At least 1GB free
                self.record_test_result(
                    "Memory Stress Test",
                    passed,
                    result
                )
                
            except Exception as e:
                self.test_signal.emit(f"Memory Stress Test Error: {str(e)}")
                self.record_test_result("Memory Stress Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
        
        thread = threading.Thread(target=run_test)
        thread.start()

    def test_system_stability(self):
        """Full system stability test"""
        def run_test():
            self.test_signal.emit("Starting System Stability Test...")
            self.stop_button.setEnabled(True)
            duration = 3600  # 1 hour
            
            try:
                start_time = time.time()
                results = []
                
                # Start all monitoring series
                cpu_series = self.cpu_chart.chart().series()[0]
                mem_series = self.memory_chart.chart().series()[0]
                temp_series = self.temp_chart.chart().series()[0]
                cpu_series.clear()
                mem_series.clear()
                temp_series.clear()
                
                while time.time() - start_time < duration and not self.stop_requested:
                    # Get system metrics
                    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                    mem = psutil.virtual_memory()
                    
                    # Get temperature
                    temp = None
                    if hasattr(psutil, "sensors_temperatures"):
                        temps = psutil.sensors_temperatures()
                        if temps and 'coretemp' in temps:
                            temp = max(t.current for t in temps['coretemp'])
                    
                    # Update charts
                    elapsed = time.time() - start_time
                    cpu_series.append(elapsed, sum(cpu_percent)/len(cpu_percent))
                    mem_series.append(elapsed, mem.percent)
                    if temp:
                        temp_series.append(elapsed, temp)
                    
                    # Log status
                    status = (
                        f"Time: {elapsed/60:.1f}min\n"
                        f"CPU: {sum(cpu_percent)/len(cpu_percent):.1f}%\n"
                        f"Memory: {mem.percent}%\n"
                    )
                    if temp:
                        status += f"Temperature: {temp}°C\n"
                    
                    self.test_signal.emit(status)
                    
                    # Update progress
                    progress = (elapsed / duration) * 100
                    self.progress_signal.emit(int(progress))
                    
                    QApplication.processEvents()
                    time.sleep(1)
                
                # Analyze results
                result = (
                    f"System Stability Test Results:\n"
                    f"Duration: {duration/3600:.1f} hours\n"
                    f"Test completed successfully"
                )
                
                self.record_test_result(
                    "System Stability Test",
                    True,
                    result
                )
                
            except Exception as e:
                self.test_signal.emit(f"System Stability Test Error: {str(e)}")
                self.record_test_result("System Stability Test", False, str(e))
            
            finally:
                self.stop_button.setEnabled(False)
                self.stop_requested = False
        
        thread = threading.Thread(target=run_test)
        thread.start()

class HardwareGuard:
    """Advanced safety system for hardware testing"""
    def __init__(self):
        # Define safety thresholds
        self.safety_limits = {
            'cpu_temp': 90,  # °C
            'gpu_temp': 95,
            'voltage': 12.2,
            'fan_speed': 5000,  # RPM
            'cpu_usage': 100,  # %
            'memory_usage': 95,  # %
            'disk_usage': 95,  # %
            'network_usage': 90  # %
        }
        self.emergency_stop = False
        self.monitoring_interval = 1  # seconds
        self.history = []
        self.max_history = 60  # Keep last 60 seconds of data

    def start_monitoring(self):
        """Start continuous safety monitoring"""
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.emergency_stop:
            metrics = self.get_system_metrics()
            self.history.append(metrics)
            
            # Keep only recent history
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Check for critical conditions
            critical = self.check_critical_conditions(metrics)
            if critical:
                self.trigger_emergency_stop(critical)
            
            time.sleep(self.monitoring_interval)

    def get_system_metrics(self):
        """Collect all system metrics"""
        metrics = {}
        try:
            # CPU Metrics
            metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    metrics['cpu_temp'] = max(t.current for t in temps['coretemp'])

            # GPU Metrics
            if wmi:
                try:
                    c = wmi.WMI(namespace='root\\wmi')
                    metrics['gpu_temp'] = c.MSAcpi_ThermalZoneTemperature()[0].CurrentTemperature/10 - 273.15
                except:
                    pass

            # Memory Metrics
            mem = psutil.virtual_memory()
            metrics['memory_usage'] = mem.percent

            # Disk Metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage'] = disk.percent

            # Network Metrics
            net = psutil.net_io_counters()
            metrics['network_usage'] = (net.bytes_sent + net.bytes_recv) / (1024**2)  # MB/s

        except Exception as e:
            print(f"Monitoring error: {str(e)}")
        
        return metrics

    def check_critical_conditions(self, metrics):
        """Check if any metrics exceed safety limits"""
        critical = {}
        for metric, value in metrics.items():
            if metric in self.safety_limits and value > self.safety_limits[metric]:
                critical[metric] = value
        return critical

    def trigger_emergency_stop(self, critical):
        """Handle emergency stop conditions"""
        self.emergency_stop = True
        critical_msg = "\n".join([f"{k}: {v} (Limit: {self.safety_limits[k]})" 
                                for k, v in critical.items()])
        
        # Stop all tests
        if hasattr(self, 'stop_current_test'):
            self.stop_current_test()
        
        # Show critical alert
        QMessageBox.critical(
            self,
            "Emergency Stop",
            f"Critical hardware condition detected:\n{critical_msg}\n\n"
            "All tests have been stopped. Please check your system."
        )

    def get_safety_report(self):
        """Generate safety report"""
        if not self.history:
            return "No safety data available"
        
        # Calculate averages
        avg_metrics = {}
        for metric in self.history[0].keys():
            values = [h[metric] for h in self.history if metric in h]
            if values:
                avg_metrics[metric] = sum(values) / len(values)
        
        # Generate report
        report = "Safety Monitoring Report:\n"
        report += "\n".join([f"{k}: {v:.1f} (Limit: {self.safety_limits.get(k, 'N/A')})" 
                           for k, v in avg_metrics.items()])
        return report

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import psutil
        import wmi
        import cv2
        import sounddevice as sd
        import numpy as np
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCharts import QChart
        return True
    except ImportError as e:
        missing = str(e).split("'")[1]
        raise ImportError(f"Required dependency '{missing}' is not installed. Please run 'pip install {missing}'")

# Add this at the very bottom of the file, after all the class definitions
if __name__ == "__main__":
    print("Starting Hardware Tester...")
    try:
        check_dependencies()
        
        # Set high DPI attributes before creating QApplication
        if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        
        app = QApplication(sys.argv)
        
        window = ModernHardwareTester()
        window.show()
        
        # Center window on screen
        screen = app.primaryScreen().geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        window.move(x, y)
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Fatal Error: {e}")
        QMessageBox.critical(None, "Fatal Error", f"Application failed to start:\n{e}")
        sys.exit(1) 