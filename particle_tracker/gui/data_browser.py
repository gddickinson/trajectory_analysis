#!/usr/bin/env python3
"""
Enhanced Data Browser Module
============================

Advanced data browser supporting hierarchical experiment organization,
sophisticated analysis results, and integration with the enhanced analysis pipeline.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QTableView, QAbstractItemView, QPushButton, QLabel, QGroupBox,
    QProgressBar, QTextEdit, QComboBox, QCheckBox, QSplitter,
    QHeaderView, QMenu, QMessageBox, QFileDialog, QFrame, QTabWidget,
    QLineEdit, QToolButton, QScrollArea, QGridLayout, QSpinBox,
    QListWidget, QListWidgetItem, QStackedWidget, QToolBar, QSizePolicy,
    QInputDialog, QDialog
)
from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal,
    QSortFilterProxyModel, QTimer, QThread, pyqtSlot
)
from PyQt6.QtGui import (
    QFont, QAction, QIcon, QStandardItemModel, QStandardItem,
    QColor, QBrush, QPalette
)

from particle_tracker.core.data_manager import EnhancedDataManager as DataManager, DataType


class DataHierarchyItem:
    """Represents an item in the hierarchical data structure."""

    def __init__(self, name: str, item_type: str, parent=None):
        self.name = name
        self.item_type = item_type  # 'experiment', 'condition', 'file', 'analysis_result'
        self.parent = parent
        self.children = []
        self.metadata = {}
        self.data_reference = None  # Reference to actual data

        if parent:
            parent.add_child(self)

    def add_child(self, child):
        """Add a child item."""
        self.children.append(child)
        child.parent = self

    def remove_child(self, child):
        """Remove a child item."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def get_path(self) -> str:
        """Get the full hierarchical path."""
        if self.parent is None:
            return self.name
        return f"{self.parent.get_path()}/{self.name}"

    def find_child(self, name: str, item_type: str = None):
        """Find a child by name and optionally type."""
        for child in self.children:
            if child.name == name and (item_type is None or child.item_type == item_type):
                return child
        return None


class AdvancedMetadataWidget(QWidget):
    """Widget for displaying detailed metadata about selected data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the metadata display UI."""
        layout = QVBoxLayout(self)

        # Create tabs for different types of metadata
        self.tab_widget = QTabWidget()

        # Basic info tab
        self.basic_info = QTextEdit()
        self.basic_info.setReadOnly(True)
        self.basic_info.setMaximumHeight(200)
        self.tab_widget.addTab(self.basic_info, "Basic Info")

        # Analysis results tab
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        self.tab_widget.addTab(self.analysis_results, "Analysis Results")

        # Statistics tab
        self.statistics = QTextEdit()
        self.statistics.setReadOnly(True)
        self.tab_widget.addTab(self.statistics, "Statistics")

        # Classification tab
        self.classification = QTextEdit()
        self.classification.setReadOnly(True)
        self.tab_widget.addTab(self.classification, "Classification")

        layout.addWidget(self.tab_widget)

    def update_metadata(self, item: DataHierarchyItem, data_manager: DataManager):
        """Update metadata display for the selected item."""
        self._update_basic_info(item, data_manager)
        self._update_analysis_results(item, data_manager)
        self._update_statistics(item, data_manager)
        self._update_classification(item, data_manager)

    def _update_basic_info(self, item: DataHierarchyItem, data_manager: DataManager):
        """Update basic information display."""
        info_lines = []
        info_lines.append(f"<b>Name:</b> {item.name}")
        info_lines.append(f"<b>Type:</b> {item.item_type}")
        info_lines.append(f"<b>Path:</b> {item.get_path()}")

        if item.data_reference:
            data = data_manager.get_data(item.data_reference)
            data_info = data_manager.get_data_info(item.data_reference)

            if data_info:
                info_lines.append(f"<b>Data Type:</b> {data_info.data_type}")
                info_lines.append(f"<b>Shape:</b> {data_info.shape}")
                info_lines.append(f"<b>Data Type:</b> {data_info.dtype}")

                if data_info.n_tracks:
                    info_lines.append(f"<b>Number of Tracks:</b> {data_info.n_tracks}")
                if data_info.n_frames:
                    info_lines.append(f"<b>Number of Frames:</b> {data_info.n_frames}")
                if data_info.file_path:
                    info_lines.append(f"<b>Source File:</b> {data_info.file_path}")

        # Add metadata from the item
        if item.metadata:
            info_lines.append("<br><b>Additional Metadata:</b>")
            for key, value in item.metadata.items():
                info_lines.append(f"<b>{key}:</b> {value}")

        self.basic_info.setHtml("<br>".join(info_lines))

    def _update_analysis_results(self, item: DataHierarchyItem, data_manager: DataManager):
        """Update analysis results display."""
        if not item.data_reference:
            self.analysis_results.setHtml("No analysis data available")
            return

        data = data_manager.get_data(item.data_reference)
        if not isinstance(data, pd.DataFrame):
            self.analysis_results.setHtml("Data is not in tabular format")
            return

        results_lines = []

        # Check for various analysis results
        analysis_columns = {
            'radius_gyration': 'Radius of Gyration',
            'sRg': 'Scaled Radius of Gyration',
            'asymmetry': 'Asymmetry',
            'skewness': 'Skewness',
            'kurtosis': 'Kurtosis',
            'fracDimension': 'Fractal Dimension',
            'eigenvalue_ratio': 'Eigenvalue Ratio',
            'step_alignment': 'Step Alignment',
            'directionality_ratio': 'Directionality Ratio',
            'velocity': 'Velocity',
            'diffusion_coefficient': 'Diffusion Coefficient',
            'mobility_classification': 'Mobility Classification',
            'linear_classification': 'Linearity Classification',
            'SVM': 'SVM Classification'
        }

        available_analyses = []
        for col, name in analysis_columns.items():
            if col in data.columns:
                available_analyses.append(name)

        if available_analyses:
            results_lines.append("<b>Available Analysis Results:</b>")
            results_lines.extend([f"• {analysis}" for analysis in available_analyses])
        else:
            results_lines.append("No recognized analysis results found")

        # Check for density analysis
        density_cols = [col for col in data.columns if 'nnCountInFrame_within_' in col]
        if density_cols:
            results_lines.append("<br><b>Density Analysis:</b>")
            radii = [col.split('_')[-2] for col in density_cols]
            results_lines.append(f"• Neighbor counts at radii: {', '.join(radii)} pixels")

        # Check for background subtraction
        bg_cols = [col for col in data.columns if 'roi' in col.lower() or 'background' in col.lower()]
        if bg_cols:
            results_lines.append("<br><b>Background Analysis:</b>")
            results_lines.extend([f"• {col}" for col in bg_cols])

        self.analysis_results.setHtml("<br>".join(results_lines))

    def _update_statistics(self, item: DataHierarchyItem, data_manager: DataManager):
        """Update statistics display."""
        if not item.data_reference:
            self.statistics.setHtml("No data available for statistics")
            return

        data = data_manager.get_data(item.data_reference)
        if not isinstance(data, pd.DataFrame):
            self.statistics.setHtml("Data is not in tabular format")
            return

        stats_lines = []

        # Basic statistics
        stats_lines.append(f"<b>Total Rows:</b> {len(data)}")

        if 'track_number' in data.columns:
            n_tracks = data['track_number'].nunique()
            track_lengths = data.groupby('track_number').size()
            stats_lines.append(f"<b>Number of Tracks:</b> {n_tracks}")
            stats_lines.append(f"<b>Mean Track Length:</b> {track_lengths.mean():.1f}")
            stats_lines.append(f"<b>Track Length Range:</b> {track_lengths.min()}-{track_lengths.max()}")

        # Classification statistics
        if 'mobility_classification' in data.columns:
            mobility_counts = data.groupby('track_number')['mobility_classification'].first().value_counts()
            stats_lines.append("<br><b>Mobility Classification:</b>")
            for classification, count in mobility_counts.items():
                pct = (count / len(mobility_counts)) * 100
                stats_lines.append(f"• {classification}: {count} ({pct:.1f}%)")

        if 'linear_classification' in data.columns:
            linear_counts = data.groupby('track_number')['linear_classification'].first().value_counts()
            stats_lines.append("<br><b>Linearity Classification:</b>")
            for classification, count in linear_counts.items():
                pct = (count / len(linear_counts)) * 100
                stats_lines.append(f"• {classification}: {count} ({pct:.1f}%)")

        if 'SVM' in data.columns:
            svm_counts = data.groupby('track_number')['SVM'].first().value_counts()
            stats_lines.append("<br><b>SVM Classification:</b>")
            for classification, count in svm_counts.items():
                pct = (count / len(svm_counts)) * 100
                stats_lines.append(f"• Class {classification}: {count} ({pct:.1f}%)")

        # Numerical statistics for key metrics
        numerical_metrics = ['radius_gyration', 'sRg', 'velocity', 'diffusion_coefficient']
        for metric in numerical_metrics:
            if metric in data.columns:
                values = data.groupby('track_number')[metric].first().dropna()
                if len(values) > 0:
                    stats_lines.append(f"<br><b>{metric}:</b>")
                    stats_lines.append(f"• Mean: {values.mean():.3f}")
                    stats_lines.append(f"• Std: {values.std():.3f}")
                    stats_lines.append(f"• Range: {values.min():.3f} - {values.max():.3f}")

        self.statistics.setHtml("<br>".join(stats_lines))

    def _update_classification(self, item: DataHierarchyItem, data_manager: DataManager):
        """Update classification display with detailed breakdown."""
        if not item.data_reference:
            self.classification.setHtml("No classification data available")
            return

        data = data_manager.get_data(item.data_reference)
        if not isinstance(data, pd.DataFrame):
            self.classification.setHtml("Data is not in tabular format")
            return

        class_lines = []

        # Detailed mobility classification
        if 'mobility_classification' in data.columns:
            mobility_data = data.groupby('track_number')['mobility_classification'].first()
            total_tracks = len(mobility_data)

            class_lines.append("<b>Mobility Classification:</b>")
            for classification in ['mobile', 'immobile']:
                count = (mobility_data == classification).sum()
                pct = (count / total_tracks) * 100 if total_tracks > 0 else 0
                class_lines.append(f"• {classification.title()}: {count}/{total_tracks} ({pct:.1f}%)")

        # Detailed linearity classification
        if 'linear_classification' in data.columns:
            linear_data = data.groupby('track_number')['linear_classification'].first()

            class_lines.append("<br><b>Linearity Classification:</b>")

            # Group linear classifications
            linear_types = ['linear_unidirectional', 'linear_bidirectional', 'linear']
            non_linear_count = (linear_data == 'non_linear').sum()
            linear_count = sum((linear_data == lt).sum() for lt in linear_types)

            total_classified = linear_count + non_linear_count
            if total_classified > 0:
                linear_pct = (linear_count / total_classified) * 100
                nonlinear_pct = (non_linear_count / total_classified) * 100

                class_lines.append(f"• Linear: {linear_count}/{total_classified} ({linear_pct:.1f}%)")
                class_lines.append(f"• Non-linear: {non_linear_count}/{total_classified} ({nonlinear_pct:.1f}%)")

                # Breakdown of linear types
                if linear_count > 0:
                    class_lines.append("<br><b>Linear Subtypes:</b>")
                    for lt in linear_types:
                        count = (linear_data == lt).sum()
                        if count > 0:
                            pct = (count / linear_count) * 100
                            class_lines.append(f"• {lt.replace('_', ' ').title()}: {count}/{linear_count} ({pct:.1f}%)")

        # SVM classification with confidence if available
        if 'SVM' in data.columns:
            svm_data = data.groupby('track_number')['SVM'].first()

            class_lines.append("<br><b>SVM Classification:</b>")
            svm_mapping = {1: 'Mobile', 2: 'Confined', 3: 'Trapped'}

            for svm_class, label in svm_mapping.items():
                count = (svm_data == svm_class).sum()
                total = len(svm_data)
                pct = (count / total) * 100 if total > 0 else 0
                class_lines.append(f"• {label} (Class {svm_class}): {count}/{total} ({pct:.1f}%)")

            # Add confidence information if available
            if 'SVM_confidence' in data.columns:
                confidence_data = data.groupby('track_number')['SVM_confidence'].first()
                mean_confidence = confidence_data.mean()
                class_lines.append(f"<br><b>Mean SVM Confidence:</b> {mean_confidence:.3f}")

        self.classification.setHtml("<br>".join(class_lines))


class HierarchicalDataTreeWidget(QTreeWidget):
    """Enhanced tree widget for hierarchical data organization."""

    itemSelected = pyqtSignal(object)  # DataHierarchyItem
    itemDoubleClicked = pyqtSignal(object)  # DataHierarchyItem

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Store mapping of QTreeWidgetItem to DataHierarchyItem
        self.item_mapping = {}

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the tree widget UI."""
        self.setHeaderLabels(["Name", "Type", "Tracks", "Files", "Status"])
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Enable drag and drop for reorganization
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

    def _connect_signals(self):
        """Connect signals."""
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    def add_hierarchy_item(self, item: DataHierarchyItem, parent_tree_item=None):
        """Add a DataHierarchyItem to the tree."""
        if parent_tree_item is None:
            tree_item = QTreeWidgetItem(self)
        else:
            tree_item = QTreeWidgetItem(parent_tree_item)

        self._update_tree_item(tree_item, item)
        self.item_mapping[tree_item] = item

        # Add children recursively
        for child in item.children:
            self.add_hierarchy_item(child, tree_item)

        return tree_item

    def _update_tree_item(self, tree_item: QTreeWidgetItem, data_item: DataHierarchyItem):
        """Update tree item display based on data item."""
        tree_item.setText(0, data_item.name)
        tree_item.setText(1, data_item.item_type.replace('_', ' ').title())

        # Calculate statistics for display
        n_tracks = self._count_tracks_in_hierarchy(data_item)
        n_files = self._count_files_in_hierarchy(data_item)

        tree_item.setText(2, str(n_tracks) if n_tracks > 0 else "-")
        tree_item.setText(3, str(n_files) if n_files > 0 else "-")

        # Set status based on data availability
        if data_item.data_reference:
            tree_item.setText(4, "Loaded")
            tree_item.setForeground(4, QBrush(QColor(0, 150, 0)))
        elif data_item.children:
            tree_item.setText(4, "Container")
            tree_item.setForeground(4, QBrush(QColor(0, 100, 200)))
        else:
            tree_item.setText(4, "Empty")
            tree_item.setForeground(4, QBrush(QColor(150, 150, 150)))

        # Set icon based on type
        self._set_item_icon(tree_item, data_item.item_type)

    def _count_tracks_in_hierarchy(self, item: DataHierarchyItem) -> int:
        """Count total tracks in this item and all children."""
        total = 0

        # If this item has data, count its tracks
        if hasattr(item, 'metadata') and 'n_tracks' in item.metadata:
            total += item.metadata['n_tracks']

        # Recursively count children
        for child in item.children:
            total += self._count_tracks_in_hierarchy(child)

        return total

    def _count_files_in_hierarchy(self, item: DataHierarchyItem) -> int:
        """Count total files in this item and all children."""
        if item.item_type == 'file':
            return 1
        return sum(self._count_files_in_hierarchy(child) for child in item.children)

    def _set_item_icon(self, tree_item: QTreeWidgetItem, item_type: str):
        """Set appropriate icon for the item type."""
        # This could be enhanced with actual icons
        pass

    def _on_item_clicked(self, tree_item: QTreeWidgetItem, column: int):
        """Handle item click."""
        if tree_item in self.item_mapping:
            data_item = self.item_mapping[tree_item]
            self.itemSelected.emit(data_item)

    def _on_item_double_clicked(self, tree_item: QTreeWidgetItem, column: int):
        """Handle item double click."""
        if tree_item in self.item_mapping:
            data_item = self.item_mapping[tree_item]
            self.itemDoubleClicked.emit(data_item)

    def _show_context_menu(self, position):
        """Show context menu for tree items."""
        tree_item = self.itemAt(position)
        if tree_item is None:
            return

        data_item = self.item_mapping.get(tree_item)
        if data_item is None:
            return

        menu = QMenu(self)

        # Context-specific actions based on item type
        if data_item.item_type == 'experiment':
            self._add_experiment_actions(menu, data_item)
        elif data_item.item_type == 'condition':
            self._add_condition_actions(menu, data_item)
        elif data_item.item_type == 'file':
            self._add_file_actions(menu, data_item)
        elif data_item.item_type == 'analysis_result':
            self._add_analysis_actions(menu, data_item)

        # Common actions
        menu.addSeparator()

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_item(data_item, tree_item))
        menu.addAction(rename_action)

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_item(data_item, tree_item))
        menu.addAction(delete_action)

        menu.exec(self.mapToGlobal(position))

    def _add_experiment_actions(self, menu: QMenu, item: DataHierarchyItem):
        """Add experiment-specific actions to context menu."""
        add_condition_action = QAction("Add Condition", self)
        add_condition_action.triggered.connect(lambda: self._add_condition(item))
        menu.addAction(add_condition_action)

        import_data_action = QAction("Import Data Folder", self)
        import_data_action.triggered.connect(lambda: self._import_data_folder(item))
        menu.addAction(import_data_action)

    def _add_condition_actions(self, menu: QMenu, item: DataHierarchyItem):
        """Add condition-specific actions to context menu."""
        add_file_action = QAction("Add File", self)
        add_file_action.triggered.connect(lambda: self._add_file(item))
        menu.addAction(add_file_action)

        batch_process_action = QAction("Batch Process", self)
        batch_process_action.triggered.connect(lambda: self._batch_process_condition(item))
        menu.addAction(batch_process_action)

    def _add_file_actions(self, menu: QMenu, item: DataHierarchyItem):
        """Add file-specific actions to context menu."""
        analyze_action = QAction("Analyze", self)
        analyze_action.triggered.connect(lambda: self._analyze_file(item))
        menu.addAction(analyze_action)

        export_action = QAction("Export", self)
        export_action.triggered.connect(lambda: self._export_file(item))
        menu.addAction(export_action)

    def _add_analysis_actions(self, menu: QMenu, item: DataHierarchyItem):
        """Add analysis result-specific actions to context menu."""
        view_action = QAction("View Results", self)
        view_action.triggered.connect(lambda: self._view_analysis_results(item))
        menu.addAction(view_action)

        export_action = QAction("Export Results", self)
        export_action.triggered.connect(lambda: self._export_analysis_results(item))
        menu.addAction(export_action)

    def _rename_item(self, data_item: DataHierarchyItem, tree_item: QTreeWidgetItem):
        """Rename an item."""
        new_name, ok = QInputDialog.getText(
            self, "Rename Item", "Enter new name:", text=data_item.name
        )
        if ok and new_name and new_name != data_item.name:
            data_item.name = new_name
            self._update_tree_item(tree_item, data_item)

    def _delete_item(self, data_item: DataHierarchyItem, tree_item: QTreeWidgetItem):
        """Delete an item."""
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete '{data_item.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if data_item.parent:
                data_item.parent.remove_child(data_item)
            self.takeTopLevelItem(self.indexOfTopLevelItem(tree_item))

    # Placeholder methods for actions - these would be implemented with actual functionality
    def _add_condition(self, experiment_item: DataHierarchyItem):
        """Add a new condition to an experiment."""
        name, ok = QInputDialog.getText(self, "Add Condition", "Condition name:")
        if ok and name:
            condition_item = DataHierarchyItem(name, 'condition', experiment_item)
            # Find the tree item and add child
            # Implementation would depend on specific requirements
            pass

    def _import_data_folder(self, experiment_item: DataHierarchyItem):
        """Import data from a folder."""
        # Implementation would open file dialog and process folder
        pass

    def _add_file(self, condition_item: DataHierarchyItem):
        """Add a file to a condition."""
        # Implementation would open file dialog
        pass

    def _batch_process_condition(self, condition_item: DataHierarchyItem):
        """Process all files in a condition."""
        # Implementation would trigger batch processing
        pass

    def _analyze_file(self, file_item: DataHierarchyItem):
        """Analyze a specific file."""
        # Implementation would trigger analysis
        pass

    def _export_file(self, file_item: DataHierarchyItem):
        """Export a file."""
        # Implementation would handle export
        pass

    def _view_analysis_results(self, analysis_item: DataHierarchyItem):
        """View analysis results."""
        # Implementation would open results viewer
        pass

    def _export_analysis_results(self, analysis_item: DataHierarchyItem):
        """Export analysis results."""
        # Implementation would handle export
        pass


class EnhancedDataBrowserWidget(QWidget):
    """Enhanced data browser supporting hierarchical organization and advanced analysis results."""

    dataSelected = pyqtSignal(str)  # data_name
    dataDoubleClicked = pyqtSignal(str)  # data_name
    hierarchyItemSelected = pyqtSignal(object)  # DataHierarchyItem

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.selected_data_name = None
        self.selected_hierarchy_item = None

        # Hierarchical data structure
        self.root_items = []  # List of top-level DataHierarchyItem objects

        # Timer for memory usage updates
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self._update_memory_display)
        self.memory_timer.start(5000)  # Update every 5 seconds

        self._setup_ui()
        self._connect_signals()

        self.logger.info("Enhanced data browser initialized")

    def _setup_ui(self):
        """Setup the enhanced UI."""
        layout = QVBoxLayout(self)

        # Create toolbar
        self.toolbar = self._create_toolbar()
        layout.addWidget(self.toolbar)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)

        # Left panel: hierarchical tree view
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)

        # Right panel: metadata and details
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([400, 300])

        # Status area
        status_layout = QHBoxLayout()

        # Memory usage label
        self.memory_label = QLabel("Memory usage: 0 MB")
        self.memory_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        status_layout.addWidget(self.memory_label)

        status_layout.addStretch()

        # Data count label
        self.data_count_label = QLabel("No data loaded")
        self.data_count_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        status_layout.addWidget(self.data_count_label)

        layout.addLayout(status_layout)

    def _create_toolbar(self) -> QToolBar:
        """Create the toolbar with common actions."""
        toolbar = QToolBar()
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        # New experiment action
        new_exp_action = QAction("New Experiment", self)
        new_exp_action.triggered.connect(self._new_experiment)
        toolbar.addAction(new_exp_action)

        # Import data action
        import_action = QAction("Import Data", self)
        import_action.triggered.connect(self._import_data)
        toolbar.addAction(import_action)

        toolbar.addSeparator()

        # Refresh action
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self._refresh_data)
        toolbar.addAction(refresh_action)

        # Search box
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Search:"))

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search data...")
        self.search_box.textChanged.connect(self._filter_data)
        toolbar.addWidget(self.search_box)

        return toolbar

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with hierarchical tree view."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # View mode selector
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View:"))

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Hierarchical", "Flat List", "By Type"])
        self.view_mode_combo.currentTextChanged.connect(self._change_view_mode)
        view_layout.addWidget(self.view_mode_combo)

        view_layout.addStretch()
        layout.addLayout(view_layout)

        # Main data tree
        self.data_tree = HierarchicalDataTreeWidget()
        self.data_tree.itemSelected.connect(self._on_hierarchy_item_selected)
        self.data_tree.itemDoubleClicked.connect(self._on_hierarchy_item_double_clicked)
        layout.addWidget(self.data_tree)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with metadata and details."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Details header
        self.details_header = QLabel("Select an item to view details")
        self.details_header.setStyleSheet("QLabel { font-weight: bold; padding: 5px; }")
        layout.addWidget(self.details_header)

        # Metadata widget
        self.metadata_widget = AdvancedMetadataWidget()
        layout.addWidget(self.metadata_widget)

        return panel

    def _connect_signals(self):
        """Connect signals."""
        self.data_manager.dataLoaded.connect(self._on_data_loaded)
        self.data_manager.dataRemoved.connect(self._on_data_removed)

    def _new_experiment(self):
        """Create a new experiment."""
        name, ok = QInputDialog.getText(self, "New Experiment", "Experiment name:")
        if ok and name:
            experiment_item = DataHierarchyItem(name, 'experiment')
            self.root_items.append(experiment_item)
            self.data_tree.add_hierarchy_item(experiment_item)
            self._update_data_count()

    def _import_data(self):
        """Import data files or folders."""
        options = QMessageBox()
        options.setWindowTitle("Import Data")
        options.setText("What would you like to import?")

        file_button = options.addButton("Single File", QMessageBox.ButtonRole.ActionRole)
        folder_button = options.addButton("Folder (Condition)", QMessageBox.ButtonRole.ActionRole)
        exp_folder_button = options.addButton("Experiment Folder", QMessageBox.ButtonRole.ActionRole)
        options.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

        options.exec()

        if options.clickedButton() == file_button:
            self._import_single_file()
        elif options.clickedButton() == folder_button:
            self._import_condition_folder()
        elif options.clickedButton() == exp_folder_button:
            self._import_experiment_folder()

    def _import_single_file(self):
        """Import a single data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Data File", "",
            "Data Files (*.csv *.xlsx *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            # Load the file through data manager
            success = self.data_manager.load_file(file_path)
            if success:
                # Create hierarchy structure if needed
                self._organize_loaded_data(file_path)

    def _import_condition_folder(self):
        """Import all files from a condition folder."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Condition Folder")
        if folder_path:
            folder = Path(folder_path)
            condition_name = folder.name

            # Create condition item
            condition_item = DataHierarchyItem(condition_name, 'condition')

            # Find files to import
            data_files = []
            for pattern in ['*.csv', '*.xlsx', '*.tif', '*.tiff']:
                data_files.extend(folder.glob(pattern))

            if data_files:
                # Import each file
                for file_path in data_files:
                    success = self.data_manager.load_file(str(file_path))
                    if success:
                        file_item = DataHierarchyItem(file_path.name, 'file', condition_item)
                        file_item.data_reference = file_path.stem

                if condition_item.children:
                    self.root_items.append(condition_item)
                    self.data_tree.add_hierarchy_item(condition_item)

            self._update_data_count()

    def _import_experiment_folder(self):
        """Import an entire experiment folder with conditions."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Experiment Folder")
        if folder_path:
            folder = Path(folder_path)
            experiment_name = folder.name

            # Create experiment item
            experiment_item = DataHierarchyItem(experiment_name, 'experiment')

            # Look for condition subdirectories
            for condition_folder in folder.iterdir():
                if condition_folder.is_dir():
                    condition_item = DataHierarchyItem(condition_folder.name, 'condition', experiment_item)

                    # Import files from condition folder
                    data_files = []
                    for pattern in ['*.csv', '*.xlsx', '*.tif', '*.tiff']:
                        data_files.extend(condition_folder.glob(pattern))

                    for file_path in data_files:
                        success = self.data_manager.load_file(str(file_path))
                        if success:
                            file_item = DataHierarchyItem(file_path.name, 'file', condition_item)
                            file_item.data_reference = file_path.stem

            if experiment_item.children:
                self.root_items.append(experiment_item)
                self.data_tree.add_hierarchy_item(experiment_item)

            self._update_data_count()

    def _organize_loaded_data(self, file_path: str):
        """Organize newly loaded data into hierarchy."""
        file_path = Path(file_path)
        data_name = file_path.stem

        # Create a simple file item if no experiment structure exists
        file_item = DataHierarchyItem(file_path.name, 'file')
        file_item.data_reference = data_name

        # Add metadata
        data_info = self.data_manager.get_data_info(data_name)
        if data_info:
            file_item.metadata.update({
                'file_path': str(file_path),
                'data_type': data_info.data_type,
                'n_tracks': data_info.n_tracks,
                'n_frames': data_info.n_frames
            })

        self.root_items.append(file_item)
        self.data_tree.add_hierarchy_item(file_item)
        self._update_data_count()

    def _refresh_data(self):
        """Refresh the data view."""
        self.data_tree.clear()
        self.data_tree.item_mapping.clear()

        # Rebuild tree from root items
        for root_item in self.root_items:
            self.data_tree.add_hierarchy_item(root_item)

    def _filter_data(self, search_text: str):
        """Filter data based on search text."""
        # Simple implementation - could be enhanced with more sophisticated filtering
        for i in range(self.data_tree.topLevelItemCount()):
            item = self.data_tree.topLevelItem(i)
            self._filter_tree_item(item, search_text.lower())

    def _filter_tree_item(self, item: QTreeWidgetItem, search_text: str):
        """Recursively filter tree items."""
        if not search_text:
            item.setHidden(False)
            for i in range(item.childCount()):
                self._filter_tree_item(item.child(i), search_text)
            return

        # Check if item matches search
        matches = search_text in item.text(0).lower()

        # Check children
        any_child_visible = False
        for i in range(item.childCount()):
            child = item.child(i)
            self._filter_tree_item(child, search_text)
            if not child.isHidden():
                any_child_visible = True

        # Hide item if it doesn't match and no children are visible
        item.setHidden(not (matches or any_child_visible))

    def _change_view_mode(self, mode: str):
        """Change the view mode of the data tree."""
        # Implementation would depend on specific requirements
        # Could show data organized by type, chronologically, etc.
        pass

    def _on_data_loaded(self, data_name: str, data: Any):
        """Handle new data being loaded."""
        self._update_memory_display()
        self._update_data_count()

    def _on_data_removed(self, data_name: str):
        """Handle data being removed."""
        # Remove from hierarchy
        self._remove_data_from_hierarchy(data_name)
        self._update_memory_display()
        self._update_data_count()

    def _remove_data_from_hierarchy(self, data_name: str):
        """Remove data reference from hierarchy."""
        def remove_recursive(items):
            for item in items:
                if item.data_reference == data_name:
                    if item.parent:
                        item.parent.remove_child(item)
                    else:
                        self.root_items.remove(item)
                    return True
                if remove_recursive(item.children):
                    return True
            return False

        remove_recursive(self.root_items)
        self._refresh_data()

    def _on_hierarchy_item_selected(self, item: DataHierarchyItem):
        """Handle hierarchy item selection."""
        self.selected_hierarchy_item = item
        self.details_header.setText(f"Details: {item.name} ({item.item_type})")

        # Update metadata display
        self.metadata_widget.update_metadata(item, self.data_manager)

        # Emit signals
        self.hierarchyItemSelected.emit(item)
        if item.data_reference:
            self.selected_data_name = item.data_reference
            self.dataSelected.emit(item.data_reference)

    def _on_hierarchy_item_double_clicked(self, item: DataHierarchyItem):
        """Handle hierarchy item double click."""
        if item.data_reference:
            self.dataDoubleClicked.emit(item.data_reference)

    def _update_memory_display(self):
        """Update memory usage display."""
        try:
            if hasattr(self.data_manager, 'get_memory_usage'):
                usage = self.data_manager.get_memory_usage()
                total_bytes = sum(usage.values())
                total_mb = total_bytes / (1024 * 1024)
                self.memory_label.setText(f"Memory usage: {total_mb:.1f} MB")
            else:
                # Fallback - estimate memory usage
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_label.setText(f"Memory usage: {memory_mb:.1f} MB")
        except Exception as e:
            # Fallback for any memory calculation issues
            self.memory_label.setText("Memory usage: N/A")

    def _update_data_count(self):
        """Update data count display."""
        total_files = 0
        total_experiments = 0
        total_conditions = 0

        def count_recursive(items):
            nonlocal total_files, total_experiments, total_conditions
            for item in items:
                if item.item_type == 'file':
                    total_files += 1
                elif item.item_type == 'experiment':
                    total_experiments += 1
                elif item.item_type == 'condition':
                    total_conditions += 1
                count_recursive(item.children)

        count_recursive(self.root_items)

        self.data_count_label.setText(
            f"Experiments: {total_experiments}, Conditions: {total_conditions}, Files: {total_files}"
        )

    # Public interface methods
    def get_selected_data_name(self) -> Optional[str]:
        """Get the name of the currently selected data."""
        return self.selected_data_name

    def get_selected_data(self) -> Optional[Any]:
        """Get the currently selected data."""
        if self.selected_data_name:
            return self.data_manager.get_data(self.selected_data_name)
        return None

    def get_selected_hierarchy_item(self) -> Optional[DataHierarchyItem]:
        """Get the currently selected hierarchy item."""
        return self.selected_hierarchy_item

    def add_analysis_result(self, parent_item_name: str, result_name: str, data_reference: str):
        """Add an analysis result to the hierarchy."""
        # Find parent item
        parent_item = None

        def find_item(items, name):
            for item in items:
                if item.name == name or item.data_reference == name:
                    return item
                result = find_item(item.children, name)
                if result:
                    return result
            return None

        parent_item = find_item(self.root_items, parent_item_name)
        if parent_item:
            result_item = DataHierarchyItem(result_name, 'analysis_result', parent_item)
            result_item.data_reference = data_reference

            # Add metadata about the analysis
            data_info = self.data_manager.get_data_info(data_reference)
            if data_info:
                result_item.metadata.update({
                    'created': datetime.now().isoformat(),
                    'data_type': data_info.data_type,
                    'n_tracks': data_info.n_tracks
                })

            self._refresh_data()
