#!/usr/bin/env python3
"""
Additional GUI Components
========================

Data browser
"""

import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QTableView, QAbstractItemView, QPushButton, QLabel, QGroupBox,
    QProgressBar, QTextEdit, QComboBox, QCheckBox, QSplitter,
    QHeaderView, QMenu, QMessageBox, QFileDialog, QFrame
)
from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal,
    QSortFilterProxyModel, QTimer
)
from PyQt6.QtGui import QFont, QAction, QIcon, QStandardItemModel, QStandardItem

from particle_tracker.core.data_manager import DataManager, DataType
from particle_tracker.core.analysis_engine import AnalysisEngine, AnalysisStep


class DataBrowserWidget(QWidget):
    """Widget for browsing and managing loaded data."""

    dataSelected = pyqtSignal(str)  # data_name
    dataDoubleClicked = pyqtSignal(str)  # data_name

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.selected_data_name = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Data tree widget
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderLabels(["Name", "Type", "Size", "Info"])
        self.data_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.data_tree.customContextMenuRequested.connect(self._show_context_menu)
        self.data_tree.itemClicked.connect(self._on_item_clicked)
        self.data_tree.itemDoubleClicked.connect(self._on_item_double_clicked)

        layout.addWidget(self.data_tree)

        # Memory usage label
        self.memory_label = QLabel("Memory usage: 0 MB")
        self.memory_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        layout.addWidget(self.memory_label)

        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._update_memory_display)
        self.refresh_timer.start(5000)  # Update every 5 seconds

    def _connect_signals(self):
        """Connect signals."""
        self.data_manager.dataLoaded.connect(self._on_data_loaded)
        self.data_manager.dataRemoved.connect(self._on_data_removed)

    def _on_data_loaded(self, data_name: str, data: Any):
        """Handle new data being loaded."""
        self._add_data_item(data_name, data)
        self._update_memory_display()

    def _on_data_removed(self, data_name: str):
        """Handle data being removed."""
        self._remove_data_item(data_name)
        self._update_memory_display()

    def _add_data_item(self, data_name: str, data: Any):
        """Add data item to the tree."""
        # Remove existing item if it exists
        self._remove_data_item(data_name)

        # Get data info
        data_info = self.data_manager.get_data_info(data_name)

        # Create tree item
        item = QTreeWidgetItem()
        item.setText(0, data_name)
        item.setText(1, data_info.data_type if data_info else "unknown")
        item.setText(2, self._format_size(data))
        item.setText(3, self._format_info(data, data_info))
        item.setData(0, Qt.ItemDataRole.UserRole, data_name)

        # Set icon based on data type
        # item.setIcon(0, self._get_data_icon(data_info.data_type if data_info else "unknown"))

        self.data_tree.addTopLevelItem(item)

        # Auto-resize columns
        for i in range(4):
            self.data_tree.resizeColumnToContents(i)

    def _remove_data_item(self, data_name: str):
        """Remove data item from the tree."""
        root = self.data_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == data_name:
                root.removeChild(item)
                break

    def _format_size(self, data: Any) -> str:
        """Format data size for display."""
        if isinstance(data, np.ndarray):
            size_bytes = data.nbytes
        elif isinstance(data, pd.DataFrame):
            size_bytes = data.memory_usage(deep=True).sum()
        else:
            return "N/A"

        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _format_info(self, data: Any, data_info) -> str:
        """Format data info for display."""
        if data_info is None:
            return "No info"

        info_parts = []

        if data_info.shape:
            if len(data_info.shape) == 1:
                info_parts.append(f"{data_info.shape[0]} items")
            elif len(data_info.shape) == 2:
                info_parts.append(f"{data_info.shape[0]}×{data_info.shape[1]}")
            elif len(data_info.shape) == 3:
                info_parts.append(f"{data_info.shape[0]}×{data_info.shape[1]}×{data_info.shape[2]}")

        if data_info.n_tracks:
            info_parts.append(f"{data_info.n_tracks} tracks")

        if data_info.n_frames:
            info_parts.append(f"{data_info.n_frames} frames")

        return ", ".join(info_parts) if info_parts else data_info.dtype

    def _update_memory_display(self):
        """Update memory usage display."""
        usage = self.data_manager.get_memory_usage()
        total_bytes = sum(usage.values())

        # Convert to MB
        total_mb = total_bytes / (1024 * 1024)
        self.memory_label.setText(f"Memory usage: {total_mb:.1f} MB")

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click."""
        data_name = item.data(0, Qt.ItemDataRole.UserRole)
        if data_name:
            self.selected_data_name = data_name
            self.dataSelected.emit(data_name)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item double click."""
        data_name = item.data(0, Qt.ItemDataRole.UserRole)
        if data_name:
            self.dataDoubleClicked.emit(data_name)

    def _show_context_menu(self, position):
        """Show context menu for data items."""
        item = self.data_tree.itemAt(position)
        if item is None:
            return

        data_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not data_name:
            return

        menu = QMenu(self)

        # View action
        view_action = QAction("View Data", self)
        view_action.triggered.connect(lambda: self._view_data(data_name))
        menu.addAction(view_action)

        # Export action
        export_action = QAction("Export...", self)
        export_action.triggered.connect(lambda: self._export_data(data_name))
        menu.addAction(export_action)

        # Duplicate action
        duplicate_action = QAction("Duplicate", self)
        duplicate_action.triggered.connect(lambda: self._duplicate_data(data_name))
        menu.addAction(duplicate_action)

        menu.addSeparator()

        # Remove action
        remove_action = QAction("Remove", self)
        remove_action.triggered.connect(lambda: self._remove_data(data_name))
        menu.addAction(remove_action)

        menu.exec(self.data_tree.mapToGlobal(position))

    def _view_data(self, data_name: str):
        """View data in a new window."""
        data = self.data_manager.get_data(data_name)
        if isinstance(data, pd.DataFrame):
            self._show_dataframe_viewer(data_name, data)
        else:
            QMessageBox.information(self, "Info", f"Data type: {type(data).__name__}")

    def _show_dataframe_viewer(self, data_name: str, data: pd.DataFrame):
        """Show DataFrame in a viewer window."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Data Viewer - {data_name}")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # Create table view
        table_view = QTableView()
        model = PandasTableModel(data)
        table_view.setModel(model)
        layout.addWidget(table_view)

        dialog.exec()

    def _export_data(self, data_name: str):
        """Export data to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {data_name}", f"{data_name}.csv",
            "CSV Files (*.csv);;JSON Files (*.json);;Excel Files (*.xlsx);;All Files (*)"
        )

        if file_path:
            if self.data_manager.save_data(data_name, file_path):
                QMessageBox.information(self, "Success", f"Data exported to {file_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to export data")

    def _duplicate_data(self, data_name: str):
        """Duplicate data with new name."""
        from PyQt6.QtWidgets import QInputDialog

        new_name, ok = QInputDialog.getText(
            self, "Duplicate Data", "Enter new name:", text=f"{data_name}_copy"
        )

        if ok and new_name:
            if self.data_manager.duplicate_data(data_name, new_name):
                QMessageBox.information(self, "Success", f"Data duplicated as {new_name}")
            else:
                QMessageBox.warning(self, "Error", "Failed to duplicate data")

    def _remove_data(self, data_name: str):
        """Remove data after confirmation."""
        reply = QMessageBox.question(
            self, "Confirm Removal",
            f"Are you sure you want to remove '{data_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.data_manager.remove_data(data_name)

    def get_selected_data_name(self) -> Optional[str]:
        """Get the name of the currently selected data."""
        return self.selected_data_name

    def get_selected_data(self) -> Optional[Any]:
        """Get the currently selected data."""
        if self.selected_data_name:
            return self.data_manager.get_data(self.selected_data_name)
        return None

