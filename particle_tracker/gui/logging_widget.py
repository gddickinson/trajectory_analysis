#!/usr/bin/env python3
"""
Additional GUI Components
========================

Logging widgets.
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



class LoggingWidget(QTextEdit):
    """Widget for displaying log messages."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setReadOnly(True)
        self.setMaximumHeight(200)

        # Setup font
        font = QFont("Courier", 9)
        self.setFont(font)

        # Setup logging handler
        self.handler = TextEditLogHandler(self)
        self.handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.handler.setFormatter(formatter)

        # Add handler to root logger
        logging.getLogger().addHandler(self.handler)

    def closeEvent(self, event):
        """Remove handler when widget is closed."""
        logging.getLogger().removeHandler(self.handler)
        super().closeEvent(event)


class TextEditLogHandler(logging.Handler):
    """Custom logging handler that writes to a QTextEdit widget."""

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        """Emit a log record."""
        try:
            msg = self.format(record)

            # Add color based on log level
            if record.levelno >= logging.ERROR:
                color = "red"
            elif record.levelno >= logging.WARNING:
                color = "orange"
            elif record.levelno >= logging.INFO:
                color = "black"
            else:
                color = "gray"

            # Format with color
            formatted_msg = f'<span style="color: {color};">{msg}</span>'

            # Append to text edit (in main thread)
            self.text_edit.append(formatted_msg)

            # Auto-scroll to bottom
            scrollbar = self.text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

            # Limit number of lines to prevent memory issues
            if self.text_edit.document().lineCount() > 1000:
                cursor = self.text_edit.textCursor()
                cursor.movePosition(cursor.MoveOperation.Start)
                cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, 100)
                cursor.removeSelectedText()

        except Exception:
            # Ignore errors in logging handler to prevent recursion
            pass


class PandasTableModel(QAbstractTableModel):
    """Table model for displaying pandas DataFrames."""

    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._data = data.copy()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._data.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isna(value):
                return "NaN"
            elif isinstance(value, float):
                return f"{value:.4f}"
            else:
                return str(value)

        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(section)
        return QVariant()

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder):
        """Sort the data by the given column."""
        self.beginResetModel()
        column_name = self._data.columns[column]
        ascending = order == Qt.SortOrder.AscendingOrder
        self._data = self._data.sort_values(by=column_name, ascending=ascending)
        self.endResetModel()
