#!/usr/bin/env python3
"""
Enhanced Logging Widget Module
==============================

Provides advanced logging capabilities for hierarchical batch analysis workflows.
Supports categorization, filtering, progress tracking, and export functionality.
"""

import logging
import json
import csv
from typing import Optional, Dict, List, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QTableView, QAbstractItemView, QPushButton, QLabel, QGroupBox,
    QProgressBar, QTextEdit, QComboBox, QCheckBox, QSplitter,
    QHeaderView, QMenu, QMessageBox, QFileDialog, QFrame,
    QTabWidget, QToolButton, QLineEdit, QSpinBox, QScrollArea
)
from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal,
    QSortFilterProxyModel, QTimer, QThread, pyqtSlot
)
from PyQt6.QtGui import QFont, QAction, QIcon, QStandardItemModel, QStandardItem, QTextCursor


class LogLevel(Enum):
    """Enhanced log levels for analysis tracking."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    PROGRESS = "PROGRESS"
    ANALYSIS = "ANALYSIS"
    BATCH = "BATCH"
    EXPORT = "EXPORT"


class LogCategory(Enum):
    """Categories for organizing log messages."""
    SYSTEM = "System"
    DETECTION = "Detection"
    LINKING = "Linking"
    FEATURES = "Features" 
    CLASSIFICATION = "Classification"
    AUTOCORR = "Autocorrelation"
    DENSITY = "Density Analysis"
    BACKGROUND = "Background Sub"
    INTERPOLATION = "Interpolation"
    LOCALIZATION = "Loc Precision"
    BATCH = "Batch Processing"
    EXPORT = "Export"
    VISUALIZATION = "Visualization"
    DATA_MANAGEMENT = "Data Mgmt"


@dataclass
class LogEntry:
    """Structured log entry for enhanced logging."""
    timestamp: str
    level: str
    category: str
    message: str
    file_name: Optional[str] = None
    condition: Optional[str] = None
    experiment: Optional[str] = None
    step: Optional[str] = None
    progress: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return asdict(self)


class LogModel(QAbstractTableModel):
    """Table model for displaying log entries."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_entries: List[LogEntry] = []
        self.headers = [
            "Timestamp", "Level", "Category", "Message", 
            "File", "Condition", "Experiment", "Step", "Progress"
        ]
        
    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.log_entries)
    
    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.headers)
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()
            
        entry = self.log_entries[index.row()]
        column = index.column()
        
        if role == Qt.ItemDataRole.DisplayRole:
            if column == 0:
                return entry.timestamp
            elif column == 1:
                return entry.level
            elif column == 2:
                return entry.category
            elif column == 3:
                return entry.message
            elif column == 4:
                return entry.file_name or ""
            elif column == 5:
                return entry.condition or ""
            elif column == 6:
                return entry.experiment or ""
            elif column == 7:
                return entry.step or ""
            elif column == 8:
                if entry.progress is not None:
                    return f"{entry.progress:.1f}%"
                return ""
                
        elif role == Qt.ItemDataRole.ForegroundRole:
            # Color code by log level
            if entry.level == "ERROR":
                return QVariant("red")
            elif entry.level == "WARNING":
                return QVariant("orange") 
            elif entry.level == "PROGRESS":
                return QVariant("blue")
            elif entry.level == "ANALYSIS":
                return QVariant("green")
                
        return QVariant()
    
    def headerData(self, section: int, orientation: Qt.Orientation, 
                   role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return QVariant()
    
    def add_entry(self, entry: LogEntry):
        """Add a new log entry."""
        self.beginInsertRows(QModelIndex(), len(self.log_entries), len(self.log_entries))
        self.log_entries.append(entry)
        self.endInsertRows()
        
    def clear_entries(self):
        """Clear all log entries."""
        self.beginResetModel()
        self.log_entries.clear()
        self.endResetModel()
        
    def get_entries(self) -> List[LogEntry]:
        """Get all log entries."""
        return self.log_entries.copy()


class LogFilter(QSortFilterProxyModel):
    """Filter model for log entries."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level_filter = None
        self.category_filter = None
        self.experiment_filter = None
        self.condition_filter = None
        self.text_filter = ""
        
    def set_level_filter(self, level: Optional[str]):
        """Set level filter."""
        self.level_filter = level
        self.invalidateFilter()
        
    def set_category_filter(self, category: Optional[str]):
        """Set category filter."""
        self.category_filter = category
        self.invalidateFilter()
        
    def set_experiment_filter(self, experiment: Optional[str]):
        """Set experiment filter."""
        self.experiment_filter = experiment
        self.invalidateFilter()
        
    def set_condition_filter(self, condition: Optional[str]):
        """Set condition filter."""
        self.condition_filter = condition
        self.invalidateFilter()
        
    def set_text_filter(self, text: str):
        """Set text filter."""
        self.text_filter = text.lower()
        self.invalidateFilter()
        
    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        """Check if row should be accepted by filters."""
        model = self.sourceModel()
        if not model or source_row >= len(model.log_entries):
            return False
            
        entry = model.log_entries[source_row]
        
        # Level filter
        if self.level_filter and entry.level != self.level_filter:
            return False
            
        # Category filter
        if self.category_filter and entry.category != self.category_filter:
            return False
            
        # Experiment filter
        if self.experiment_filter and entry.experiment != self.experiment_filter:
            return False
            
        # Condition filter
        if self.condition_filter and entry.condition != self.condition_filter:
            return False
            
        # Text filter
        if self.text_filter:
            if (self.text_filter not in entry.message.lower() and
                self.text_filter not in (entry.file_name or "").lower()):
                return False
                
        return True


class BatchProgressWidget(QWidget):
    """Widget for tracking batch analysis progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.experiment_progress = {}
        self.condition_progress = {}
        
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        
        # Overall progress
        overall_group = QGroupBox("Overall Progress")
        overall_layout = QVBoxLayout(overall_group)
        
        self.overall_progress = QProgressBar()
        self.overall_label = QLabel("Ready")
        overall_layout.addWidget(self.overall_label)
        overall_layout.addWidget(self.overall_progress)
        
        layout.addWidget(overall_group)
        
        # Experiment progress tree
        experiment_group = QGroupBox("Experiment Progress")
        experiment_layout = QVBoxLayout(experiment_group)
        
        self.experiment_tree = QTreeWidget()
        self.experiment_tree.setHeaderLabels([
            "Item", "Progress", "Status", "Files", "Tracks", "Time"
        ])
        experiment_layout.addWidget(self.experiment_tree)
        
        layout.addWidget(experiment_group)
        
        # Current operation
        current_group = QGroupBox("Current Operation")
        current_layout = QVBoxLayout(current_group)
        
        self.current_operation = QLabel("Idle")
        self.current_progress = QProgressBar()
        self.current_details = QLabel("")
        
        current_layout.addWidget(self.current_operation)
        current_layout.addWidget(self.current_progress)
        current_layout.addWidget(self.current_details)
        
        layout.addWidget(current_group)
        
    def update_overall_progress(self, value: float, message: str = ""):
        """Update overall progress."""
        self.overall_progress.setValue(int(value))
        if message:
            self.overall_label.setText(message)
            
    def update_current_operation(self, operation: str, progress: float = 0, details: str = ""):
        """Update current operation display."""
        self.current_operation.setText(operation)
        self.current_progress.setValue(int(progress))
        if details:
            self.current_details.setText(details)
            
    def add_experiment(self, experiment_name: str):
        """Add an experiment to the progress tree."""
        item = QTreeWidgetItem([experiment_name, "0%", "Pending", "0", "0", ""])
        self.experiment_tree.addTopLevelItem(item)
        self.experiment_progress[experiment_name] = item
        
    def add_condition(self, experiment_name: str, condition_name: str):
        """Add a condition to an experiment."""
        if experiment_name in self.experiment_progress:
            parent_item = self.experiment_progress[experiment_name]
            item = QTreeWidgetItem([condition_name, "0%", "Pending", "0", "0", ""])
            parent_item.addChild(item)
            self.condition_progress[f"{experiment_name}:{condition_name}"] = item
            
    def update_experiment_progress(self, experiment_name: str, progress: float, 
                                 status: str, files_processed: int = 0, tracks_found: int = 0):
        """Update experiment progress."""
        if experiment_name in self.experiment_progress:
            item = self.experiment_progress[experiment_name]
            item.setText(1, f"{progress:.1f}%")
            item.setText(2, status)
            item.setText(3, str(files_processed))
            item.setText(4, str(tracks_found))
            item.setText(5, datetime.now().strftime("%H:%M:%S"))
            
    def update_condition_progress(self, experiment_name: str, condition_name: str,
                                progress: float, status: str, files_processed: int = 0, tracks_found: int = 0):
        """Update condition progress."""
        key = f"{experiment_name}:{condition_name}"
        if key in self.condition_progress:
            item = self.condition_progress[key]
            item.setText(1, f"{progress:.1f}%")
            item.setText(2, status)
            item.setText(3, str(files_processed))
            item.setText(4, str(tracks_found))
            item.setText(5, datetime.now().strftime("%H:%M:%S"))


class EnhancedLoggingWidget(QWidget):
    """Enhanced logging widget with filtering, categorization, and export."""
    
    # Signals
    logMessage = pyqtSignal(str, str, str, str)  # level, category, message, metadata
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Data models
        self.log_model = LogModel()
        self.filter_model = LogFilter()
        self.filter_model.setSourceModel(self.log_model)
        
        # Current context
        self.current_experiment = None
        self.current_condition = None
        self.current_file = None
        self.current_step = None
        
        self._setup_ui()
        self._setup_logging_handler()
        self._connect_signals()
        
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Log viewer tab
        log_tab = self._create_log_viewer_tab()
        self.tab_widget.addTab(log_tab, "Log Viewer")
        
        # Progress tracking tab
        progress_tab = self._create_progress_tab()
        self.tab_widget.addTab(progress_tab, "Progress")
        
        # Statistics tab
        stats_tab = self._create_statistics_tab()
        self.tab_widget.addTab(stats_tab, "Statistics")
        
        layout.addWidget(self.tab_widget)
        
    def _create_log_viewer_tab(self) -> QWidget:
        """Create the log viewer tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Filter controls
        filter_frame = self._create_filter_controls()
        layout.addWidget(filter_frame)
        
        # Log table
        self.log_table = QTableView()
        self.log_table.setModel(self.filter_model)
        self.log_table.setSortingEnabled(True)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        
        # Auto-resize columns
        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Message column
        
        layout.addWidget(self.log_table)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        button_layout.addWidget(self.clear_logs_btn)
        
        self.export_logs_btn = QPushButton("Export Logs")
        self.export_logs_btn.clicked.connect(self._export_logs)
        button_layout.addWidget(self.export_logs_btn)
        
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        button_layout.addWidget(self.auto_scroll_cb)
        
        button_layout.addStretch()
        
        # Stats summary
        self.stats_label = QLabel("Total logs: 0")
        button_layout.addWidget(self.stats_label)
        
        layout.addLayout(button_layout)
        
        return tab
        
    def _create_filter_controls(self) -> QFrame:
        """Create filter control widgets."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        # Level filter
        layout.addWidget(QLabel("Level:"))
        self.level_filter = QComboBox()
        self.level_filter.addItems(["All"] + [level.value for level in LogLevel])
        self.level_filter.currentTextChanged.connect(self._apply_filters)
        layout.addWidget(self.level_filter)
        
        # Category filter
        layout.addWidget(QLabel("Category:"))
        self.category_filter = QComboBox()
        self.category_filter.addItems(["All"] + [cat.value for cat in LogCategory])
        self.category_filter.currentTextChanged.connect(self._apply_filters)
        layout.addWidget(self.category_filter)
        
        # Experiment filter
        layout.addWidget(QLabel("Experiment:"))
        self.experiment_filter = QComboBox()
        self.experiment_filter.addItems(["All"])
        self.experiment_filter.currentTextChanged.connect(self._apply_filters)
        layout.addWidget(self.experiment_filter)
        
        # Condition filter
        layout.addWidget(QLabel("Condition:"))
        self.condition_filter = QComboBox()
        self.condition_filter.addItems(["All"])
        self.condition_filter.currentTextChanged.connect(self._apply_filters)
        layout.addWidget(self.condition_filter)
        
        # Text filter
        layout.addWidget(QLabel("Search:"))
        self.text_filter = QLineEdit()
        self.text_filter.setPlaceholderText("Search messages...")
        self.text_filter.textChanged.connect(self._apply_filters)
        layout.addWidget(self.text_filter)
        
        layout.addStretch()
        
        return frame
        
    def _create_progress_tab(self) -> QWidget:
        """Create the progress tracking tab."""
        self.progress_widget = BatchProgressWidget()
        return self.progress_widget
        
    def _create_statistics_tab(self) -> QWidget:
        """Create the statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.stats_text)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.clicked.connect(self._update_statistics)
        layout.addWidget(refresh_btn)
        
        return tab
        
    def _setup_logging_handler(self):
        """Setup the custom logging handler."""
        self.handler = EnhancedLogHandler(self)
        self.handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(self.handler)
        
    def _connect_signals(self):
        """Connect internal signals."""
        self.logMessage.connect(self._add_log_entry)
        
    def _apply_filters(self):
        """Apply current filter settings."""
        level = self.level_filter.currentText()
        if level == "All":
            level = None
        self.filter_model.set_level_filter(level)
        
        category = self.category_filter.currentText()
        if category == "All":
            category = None
        self.filter_model.set_category_filter(category)
        
        experiment = self.experiment_filter.currentText()
        if experiment == "All":
            experiment = None
        self.filter_model.set_experiment_filter(experiment)
        
        condition = self.condition_filter.currentText()
        if condition == "All":
            condition = None
        self.filter_model.set_condition_filter(condition)
        
        text = self.text_filter.text()
        self.filter_model.set_text_filter(text)
        
        # Update stats
        self._update_log_stats()
        
    @pyqtSlot(str, str, str, str)
    def _add_log_entry(self, level: str, category: str, message: str, metadata_str: str):
        """Add a log entry to the model."""
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except Exception:
            metadata = {}
            
        entry = LogEntry(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            level=level,
            category=category,
            message=message,
            file_name=metadata.get('file_name'),
            condition=metadata.get('condition'),
            experiment=metadata.get('experiment'),
            step=metadata.get('step'),
            progress=metadata.get('progress'),
            metadata=metadata
        )
        
        self.log_model.add_entry(entry)
        
        # Update filter options
        self._update_filter_options(entry)
        
        # Auto-scroll
        if self.auto_scroll_cb.isChecked():
            self.log_table.scrollToBottom()
            
        # Update stats
        self._update_log_stats()
        
    def _update_filter_options(self, entry: LogEntry):
        """Update filter dropdown options based on new entries."""
        # Update experiment filter
        if entry.experiment and entry.experiment not in [
            self.experiment_filter.itemText(i) for i in range(self.experiment_filter.count())
        ]:
            self.experiment_filter.addItem(entry.experiment)
            
        # Update condition filter
        if entry.condition and entry.condition not in [
            self.condition_filter.itemText(i) for i in range(self.condition_filter.count())
        ]:
            self.condition_filter.addItem(entry.condition)
            
    def _update_log_stats(self):
        """Update log statistics display."""
        total_logs = self.log_model.rowCount()
        visible_logs = self.filter_model.rowCount()
        
        if total_logs == visible_logs:
            self.stats_label.setText(f"Total logs: {total_logs}")
        else:
            self.stats_label.setText(f"Showing: {visible_logs} / {total_logs}")
            
    def _clear_logs(self):
        """Clear all log entries."""
        reply = QMessageBox.question(
            self, "Clear Logs",
            "Are you sure you want to clear all log entries?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.log_model.clear_entries()
            self._update_log_stats()
            
    def _export_logs(self):
        """Export log entries to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "",
            "CSV Files (*.csv);;JSON Files (*.json);;Text Files (*.txt)"
        )
        
        if not file_path:
            return
            
        try:
            entries = self.log_model.get_entries()
            
            if file_path.endswith('.csv'):
                # Export as CSV
                df = pd.DataFrame([entry.to_dict() for entry in entries])
                df.to_csv(file_path, index=False)
                
            elif file_path.endswith('.json'):
                # Export as JSON
                with open(file_path, 'w') as f:
                    json.dump([entry.to_dict() for entry in entries], f, indent=2)
                    
            else:  # .txt
                # Export as text
                with open(file_path, 'w') as f:
                    for entry in entries:
                        f.write(f"{entry.timestamp} [{entry.level}] {entry.category}: {entry.message}\n")
                        
            QMessageBox.information(self, "Export Complete", f"Logs exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting logs: {e}")
            
    def _update_statistics(self):
        """Update the statistics display."""
        entries = self.log_model.get_entries()
        
        if not entries:
            self.stats_text.setText("No log entries available.")
            return
            
        # Calculate statistics
        level_counts = {}
        category_counts = {}
        experiment_counts = {}
        condition_counts = {}
        
        for entry in entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
            
            if entry.experiment:
                experiment_counts[entry.experiment] = experiment_counts.get(entry.experiment, 0) + 1
            if entry.condition:
                condition_counts[entry.condition] = condition_counts.get(entry.condition, 0) + 1
                
        # Format statistics
        stats_lines = [
            "Logging Statistics",
            "=" * 50,
            "",
            f"Total Log Entries: {len(entries)}",
            "",
            "Log Levels:",
            "-" * 20
        ]
        
        for level, count in sorted(level_counts.items()):
            pct = (count / len(entries)) * 100
            stats_lines.append(f"{level}: {count} ({pct:.1f}%)")
            
        stats_lines.extend([
            "",
            "Categories:",
            "-" * 20
        ])
        
        for category, count in sorted(category_counts.items()):
            pct = (count / len(entries)) * 100
            stats_lines.append(f"{category}: {count} ({pct:.1f}%)")
            
        if experiment_counts:
            stats_lines.extend([
                "",
                "Experiments:",
                "-" * 20
            ])
            for exp, count in sorted(experiment_counts.items()):
                stats_lines.append(f"{exp}: {count}")
                
        if condition_counts:
            stats_lines.extend([
                "",
                "Conditions:",
                "-" * 20
            ])
            for cond, count in sorted(condition_counts.items()):
                stats_lines.append(f"{cond}: {count}")
                
        self.stats_text.setText("\n".join(stats_lines))
        
    # Public interface methods
    def set_context(self, experiment: str = None, condition: str = None, 
                   file_name: str = None, step: str = None):
        """Set the current analysis context."""
        self.current_experiment = experiment
        self.current_condition = condition  
        self.current_file = file_name
        self.current_step = step
        
    def log_analysis_start(self, step: str, details: str = ""):
        """Log the start of an analysis step."""
        self.log_message("ANALYSIS", LogCategory.SYSTEM.value, 
                        f"Starting {step}", extra={'step': step, 'details': details})
                        
    def log_analysis_complete(self, step: str, duration: float = None, results: Dict = None):
        """Log the completion of an analysis step."""
        message = f"Completed {step}"
        if duration:
            message += f" in {duration:.2f}s"
        if results:
            message += f" - {results}"
        self.log_message("ANALYSIS", LogCategory.SYSTEM.value, message, 
                        extra={'step': step, 'duration': duration, 'results': results})
                        
    def log_progress(self, step: str, progress: float, details: str = ""):
        """Log progress update."""
        message = f"{step}: {progress:.1f}%"
        if details:
            message += f" - {details}"
        self.log_message("PROGRESS", LogCategory.SYSTEM.value, message,
                        extra={'step': step, 'progress': progress})
                        
    def log_batch_start(self, experiment: str, total_files: int):
        """Log start of batch processing."""
        self.set_context(experiment=experiment)
        self.log_message("BATCH", LogCategory.BATCH.value,
                        f"Starting batch analysis for {experiment} ({total_files} files)")
        self.progress_widget.add_experiment(experiment)
        
    def log_condition_start(self, condition: str, total_files: int):
        """Log start of condition processing."""
        self.set_context(condition=condition)
        self.log_message("BATCH", LogCategory.BATCH.value,
                        f"Processing condition {condition} ({total_files} files)")
        if self.current_experiment:
            self.progress_widget.add_condition(self.current_experiment, condition)
            
    def log_file_start(self, file_name: str):
        """Log start of file processing."""
        self.set_context(file_name=file_name)
        self.log_message("INFO", LogCategory.DATA_MANAGEMENT.value,
                        f"Processing file: {file_name}")
                        
    def log_message(self, level: str, category: str, message: str, extra: Dict = None):
        """Log a message with context."""
        metadata = {
            'experiment': self.current_experiment,
            'condition': self.current_condition,
            'file_name': self.current_file,
            'step': self.current_step
        }
        
        if extra:
            metadata.update(extra)
            
        self.logMessage.emit(level, category, message, json.dumps(metadata))
        
    def update_progress(self, operation: str, progress: float, details: str = ""):
        """Update progress display."""
        self.progress_widget.update_current_operation(operation, progress, details)


class EnhancedLogHandler(logging.Handler):
    """Custom logging handler for the enhanced logging widget."""
    
    def __init__(self, logging_widget: EnhancedLoggingWidget):
        super().__init__()
        self.logging_widget = logging_widget
        self.category_mapping = {
            'detection': LogCategory.DETECTION.value,
            'linking': LogCategory.LINKING.value,
            'features': LogCategory.FEATURES.value,
            'classification': LogCategory.CLASSIFICATION.value,
            'autocorrelation': LogCategory.AUTOCORR.value,
            'density': LogCategory.DENSITY.value,
            'background': LogCategory.BACKGROUND.value,
            'interpolation': LogCategory.INTERPOLATION.value,
            'localization': LogCategory.LOCALIZATION.value,
            'batch': LogCategory.BATCH.value,
            'export': LogCategory.EXPORT.value,
            'visualization': LogCategory.VISUALIZATION.value,
            'data_manager': LogCategory.DATA_MANAGEMENT.value,
        }
        
    def emit(self, record):
        """Emit a log record."""
        try:
            # Determine category from logger name
            logger_name = record.name.lower()
            category = LogCategory.SYSTEM.value
            
            for key, cat in self.category_mapping.items():
                if key in logger_name:
                    category = cat
                    break
                    
            # Format message
            message = self.format(record)
            
            # Extract metadata from record
            metadata = getattr(record, 'metadata', {})
            
            # Emit to logging widget
            self.logging_widget.logMessage.emit(
                record.levelname, category, message, json.dumps(metadata)
            )
            
        except Exception:
            # Ignore errors in logging handler
            pass