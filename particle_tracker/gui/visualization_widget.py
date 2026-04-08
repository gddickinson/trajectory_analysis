#!/usr/bin/env python3
"""
Enhanced Visualization Widget Module
====================================

Advanced visualization capabilities for enhanced particle tracking analysis
including multi-classification overlays, density heatmaps, linearity visualization,
and autocorrelation plots.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QCheckBox, QComboBox, QGroupBox,
    QGridLayout, QColorDialog, QSpinBox, QTabWidget,
    QSplitter, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ImageView, GraphicsLayoutWidget

# Try to import colorcet, but handle if it's not available
try:
    import colorcet as cc
    COLORCET_AVAILABLE = True
except ImportError:
    COLORCET_AVAILABLE = False


class EnhancedVisualizationWidget(QWidget):
    """Enhanced visualization widget with advanced analysis visualization."""

    # Signals
    frameChanged = pyqtSignal(int)
    pointClicked = pyqtSignal(float, float)
    trackSelected = pyqtSignal(int)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager

        # Data storage
        self.image_data = None
        self.current_frame = 0
        self.tracking_data = None
        self.overlay_data = {}

        # Visualization state
        self.show_tracks = True
        self.show_localizations = True
        self.show_track_ids = False
        self.show_density_heatmap = False
        self.show_linearity_overlay = False
        self.show_mobility_overlay = True
        self.track_length_limit = 50
        self.point_size = 3

        # Enhanced visualization options
        self.color_by_feature = 'track_id'
        self.density_radius = 10
        self.heatmap_resolution = 50

        # Color schemes
        self.colormaps = self._load_colormaps()
        self.current_colormap = list(self.colormaps.keys())[0] if self.colormaps else 'gray'

        # Track colors and classification colors
        self.track_colors = {}
        self.classification_colors = {
            'mobile': (255, 0, 0, 255),      # Red
            'immobile': (0, 0, 255, 255),    # Blue
            'linear_unidirectional': (0, 255, 0, 255),     # Green
            'linear_bidirectional': (255, 255, 0, 255),    # Yellow
            'non_linear': (255, 0, 255, 255),              # Magenta
            'unclassified': (128, 128, 128, 255)           # Gray
        }

        self._setup_ui()
        self._connect_signals()

        self.logger.info("Enhanced visualization widget initialized")

    def _load_colormaps(self) -> Dict[str, Any]:
        """Load available colormaps with enhanced options."""
        colormaps = {}

        # Enhanced colormap names including scientific ones
        colormap_names = [
            'viridis', 'plasma', 'inferno', 'magma', 'gray', 'hot',
            'jet', 'coolwarm', 'seismic', 'rainbow'
        ]

        for name in colormap_names:
            try:
                if hasattr(pg.colormap, 'get'):
                    colormap = pg.colormap.get(name)
                else:
                    colormap = self._create_fallback_colormap(name)

                if colormap is not None:
                    colormaps[name] = colormap

            except Exception as e:
                self.logger.debug(f"Could not load colormap '{name}': {e}")
                colormap = self._create_fallback_colormap(name)
                if colormap is not None:
                    colormaps[name] = colormap

        # Ensure we have at least one colormap
        if not colormaps:
            colormaps['gray'] = self._create_grayscale_colormap()

        return colormaps

    def _create_fallback_colormap(self, name: str):
        """Create fallback colormaps for enhanced visualization."""
        try:
            if name == 'gray':
                return self._create_grayscale_colormap()
            elif name == 'viridis':
                return self._create_viridis_colormap()
            elif name == 'jet':
                return self._create_jet_colormap()
            elif name == 'coolwarm':
                return self._create_coolwarm_colormap()
            elif name == 'rainbow':
                return self._create_rainbow_colormap()
            else:
                return self._create_grayscale_colormap()
        except Exception:
            return self._create_grayscale_colormap()

    def _create_jet_colormap(self):
        """Create jet colormap for scientific visualization."""
        try:
            positions = np.linspace(0, 1, 5)
            colors = np.array([
                [0, 0, 128, 255],      # dark blue
                [0, 0, 255, 255],      # blue
                [0, 255, 255, 255],    # cyan
                [255, 255, 0, 255],    # yellow
                [255, 0, 0, 255]       # red
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception:
            return self._create_grayscale_colormap()

    def _create_coolwarm_colormap(self):
        """Create coolwarm colormap."""
        try:
            positions = np.linspace(0, 1, 3)
            colors = np.array([
                [59, 76, 192, 255],    # cool blue
                [221, 221, 221, 255],  # neutral gray
                [180, 4, 38, 255]      # warm red
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception:
            return self._create_grayscale_colormap()

    def _create_rainbow_colormap(self):
        """Create rainbow colormap."""
        try:
            positions = np.linspace(0, 1, 7)
            colors = np.array([
                [255, 0, 255, 255],    # magenta
                [0, 0, 255, 255],      # blue
                [0, 255, 255, 255],    # cyan
                [0, 255, 0, 255],      # green
                [255, 255, 0, 255],    # yellow
                [255, 165, 0, 255],    # orange
                [255, 0, 0, 255]       # red
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception:
            return self._create_grayscale_colormap()

    def _create_grayscale_colormap(self):
        """Create a simple grayscale colormap."""
        try:
            positions = np.linspace(0, 1, 256)
            colors = np.array([[i, i, i, 255] for i in range(256)], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception:
            return None

    def _create_viridis_colormap(self):
        """Create viridis-like colormap."""
        try:
            positions = np.linspace(0, 1, 5)
            colors = np.array([
                [68, 1, 84, 255],      # dark purple
                [59, 82, 139, 255],    # blue
                [33, 145, 140, 255],   # teal
                [94, 201, 98, 255],    # green
                [253, 231, 37, 255]    # yellow
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception:
            return self._create_grayscale_colormap()

    def _setup_ui(self):
        """Setup the enhanced user interface."""
        layout = QVBoxLayout(self)

        # Create main visualization area with tabs
        self.viz_tabs = QTabWidget()

        # Main visualization tab
        main_viz_tab = self._create_main_visualization_tab()
        self.viz_tabs.addTab(main_viz_tab, "Main Visualization")

        # Density heatmap tab
        density_tab = self._create_density_heatmap_tab()
        self.viz_tabs.addTab(density_tab, "Density Analysis")

        # Feature analysis tab
        feature_tab = self._create_feature_analysis_tab()
        self.viz_tabs.addTab(feature_tab, "Feature Analysis")

        layout.addWidget(self.viz_tabs)

        # Enhanced controls panel
        controls_panel = self._create_enhanced_controls_panel()
        layout.addWidget(controls_panel)

    def _create_main_visualization_tab(self) -> QWidget:
        """Create main visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Create graphics layout
        self.graphics_widget = GraphicsLayoutWidget()
        layout.addWidget(self.graphics_widget)

        # Create plot item
        self.image_plot = self.graphics_widget.addPlot(row=0, col=0)
        self.image_plot.setAspectLocked(True)
        self.image_plot.setLabel('left', 'Y (pixels)')
        self.image_plot.setLabel('bottom', 'X (pixels)')

        # Get the ViewBox
        self.image_view = self.image_plot.getViewBox()

        # Create image item
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)

        # Create overlay for points and tracks
        self.scatter_item = pg.ScatterPlotItem()
        self.image_plot.addItem(self.scatter_item)

        # Track lines and overlay items
        self.track_items = []
        self.overlay_items = []

        # Setup initial view
        self.image_view.scene().sigMouseClicked.connect(self._on_mouse_click)

        return tab

    def _create_density_heatmap_tab(self) -> QWidget:
        """Create density heatmap visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Create density plot
        self.density_graphics = GraphicsLayoutWidget()
        self.density_plot = self.density_graphics.addPlot(row=0, col=0)
        self.density_plot.setAspectLocked(True)
        self.density_plot.setLabel('left', 'Y (pixels)')
        self.density_plot.setLabel('bottom', 'X (pixels)')

        # Density image item
        self.density_image_item = pg.ImageItem()
        self.density_plot.addItem(self.density_image_item)

        layout.addWidget(self.density_graphics)

        # Density controls
        density_controls = QGroupBox("Density Heatmap Controls")
        density_layout = QGridLayout(density_controls)

        density_layout.addWidget(QLabel("Radius:"), 0, 0)
        self.density_radius_spin = QSpinBox()
        self.density_radius_spin.setRange(1, 50)
        self.density_radius_spin.setValue(self.density_radius)
        self.density_radius_spin.valueChanged.connect(self._update_density_radius)
        density_layout.addWidget(self.density_radius_spin, 0, 1)

        density_layout.addWidget(QLabel("Resolution:"), 0, 2)
        self.density_resolution_spin = QSpinBox()
        self.density_resolution_spin.setRange(10, 200)
        self.density_resolution_spin.setValue(self.heatmap_resolution)
        self.density_resolution_spin.valueChanged.connect(self._update_density_resolution)
        density_layout.addWidget(self.density_resolution_spin, 0, 3)

        self.update_density_btn = QPushButton("Update Density Map")
        self.update_density_btn.clicked.connect(self._update_density_heatmap)
        density_layout.addWidget(self.update_density_btn, 1, 0, 1, 4)

        layout.addWidget(density_controls)

        return tab

    def _create_feature_analysis_tab(self) -> QWidget:
        """Create feature analysis visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Feature plots area
        self.feature_graphics = GraphicsLayoutWidget()
        layout.addWidget(self.feature_graphics)

        # Feature plot controls
        feature_controls = QGroupBox("Feature Analysis Controls")
        feature_layout = QGridLayout(feature_controls)

        feature_layout.addWidget(QLabel("X Feature:"), 0, 0)
        self.x_feature_combo = QComboBox()
        feature_layout.addWidget(self.x_feature_combo, 0, 1)

        feature_layout.addWidget(QLabel("Y Feature:"), 0, 2)
        self.y_feature_combo = QComboBox()
        feature_layout.addWidget(self.y_feature_combo, 0, 3)

        feature_layout.addWidget(QLabel("Color By:"), 1, 0)
        self.color_feature_combo = QComboBox()
        feature_layout.addWidget(self.color_feature_combo, 1, 1)

        self.update_feature_plot_btn = QPushButton("Update Feature Plot")
        self.update_feature_plot_btn.clicked.connect(self._update_feature_plot)
        feature_layout.addWidget(self.update_feature_plot_btn, 1, 2, 1, 2)

        layout.addWidget(feature_controls)

        return tab

    def _create_enhanced_controls_panel(self) -> QWidget:
        """Create enhanced visualization controls panel."""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QGridLayout(display_group)

        self.show_tracks_cb = QCheckBox("Show Tracks")
        self.show_tracks_cb.setChecked(self.show_tracks)
        self.show_tracks_cb.toggled.connect(self._toggle_tracks)
        display_layout.addWidget(self.show_tracks_cb, 0, 0)

        self.show_localizations_cb = QCheckBox("Show Localizations")
        self.show_localizations_cb.setChecked(self.show_localizations)
        self.show_localizations_cb.toggled.connect(self._toggle_localizations)
        display_layout.addWidget(self.show_localizations_cb, 0, 1)

        self.show_track_ids_cb = QCheckBox("Show Track IDs")
        self.show_track_ids_cb.setChecked(self.show_track_ids)
        self.show_track_ids_cb.toggled.connect(self._toggle_track_ids)
        display_layout.addWidget(self.show_track_ids_cb, 1, 0)

        self.show_mobility_overlay_cb = QCheckBox("Mobility Overlay")
        self.show_mobility_overlay_cb.setChecked(self.show_mobility_overlay)
        self.show_mobility_overlay_cb.toggled.connect(self._toggle_mobility_overlay)
        display_layout.addWidget(self.show_mobility_overlay_cb, 1, 1)

        self.show_linearity_overlay_cb = QCheckBox("Linearity Overlay")
        self.show_linearity_overlay_cb.setChecked(self.show_linearity_overlay)
        self.show_linearity_overlay_cb.toggled.connect(self._toggle_linearity_overlay)
        display_layout.addWidget(self.show_linearity_overlay_cb, 2, 0)

        # Point size control
        display_layout.addWidget(QLabel("Point Size:"), 3, 0)
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(self.point_size)
        self.point_size_spin.valueChanged.connect(self._change_point_size)
        display_layout.addWidget(self.point_size_spin, 3, 1)

        layout.addWidget(display_group)

        # Enhanced color options group
        color_group = QGroupBox("Enhanced Color Options")
        color_layout = QGridLayout(color_group)

        color_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(list(self.colormaps.keys()))
        self.colormap_combo.setCurrentText(self.current_colormap)
        self.colormap_combo.currentTextChanged.connect(self._change_colormap)
        color_layout.addWidget(self.colormap_combo, 0, 1)

        color_layout.addWidget(QLabel("Color by:"), 1, 0)
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems([
            "Track ID", "Mobility Classification", "Linear Classification",
            "Scaled Rg", "Velocity", "Density", "Eigenvalue Ratio",
            "Step Alignment", "Diffusion Coefficient", "Distance from Origin"
        ])
        self.color_by_combo.currentTextChanged.connect(self._change_color_scheme)
        color_layout.addWidget(self.color_by_combo, 1, 1)

        # Classification legend
        legend_btn = QPushButton("Show Classification Legend")
        legend_btn.clicked.connect(self._show_classification_legend)
        color_layout.addWidget(legend_btn, 2, 0, 1, 2)

        layout.addWidget(color_group)

        # Enhanced track options group
        track_group = QGroupBox("Enhanced Track Options")
        track_layout = QGridLayout(track_group)

        track_layout.addWidget(QLabel("Track Length:"), 0, 0)
        self.track_length_spin = QSpinBox()
        self.track_length_spin.setRange(1, 1000)
        self.track_length_spin.setValue(self.track_length_limit)
        self.track_length_spin.valueChanged.connect(self._filter_by_length)
        track_layout.addWidget(self.track_length_spin, 0, 1)

        # Enhanced filtering options
        track_layout.addWidget(QLabel("Show Mobility:"), 1, 0)
        self.mobility_filter_combo = QComboBox()
        self.mobility_filter_combo.addItems(["All", "Mobile", "Immobile"])
        self.mobility_filter_combo.currentTextChanged.connect(self._filter_by_mobility)
        track_layout.addWidget(self.mobility_filter_combo, 1, 1)

        track_layout.addWidget(QLabel("Show Linearity:"), 2, 0)
        self.linearity_filter_combo = QComboBox()
        self.linearity_filter_combo.addItems([
            "All", "Linear Unidirectional", "Linear Bidirectional", "Non-linear"
        ])
        self.linearity_filter_combo.currentTextChanged.connect(self._filter_by_linearity)
        track_layout.addWidget(self.linearity_filter_combo, 2, 1)

        # Feature range filtering
        track_layout.addWidget(QLabel("Min Track Length:"), 3, 0)
        self.min_track_length_spin = QSpinBox()
        self.min_track_length_spin.setRange(1, 1000)
        self.min_track_length_spin.setValue(3)
        self.min_track_length_spin.valueChanged.connect(self._filter_by_length)
        track_layout.addWidget(self.min_track_length_spin, 3, 1)

        layout.addWidget(track_group)

        # View controls group
        view_group = QGroupBox("View Controls")
        view_layout = QHBoxLayout(view_group)

        self.zoom_fit_btn = QPushButton("Zoom Fit")
        self.zoom_fit_btn.clicked.connect(self.zoom_fit)
        view_layout.addWidget(self.zoom_fit_btn)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        view_layout.addWidget(self.reset_view_btn)

        self.export_view_btn = QPushButton("Export View")
        self.export_view_btn.clicked.connect(self.export_current_view)
        view_layout.addWidget(self.export_view_btn)

        layout.addWidget(view_group)

        layout.addStretch()

        return panel

    def _connect_signals(self):
        """Connect internal signals."""
        self.data_manager.dataLoaded.connect(self._on_data_loaded)

    def set_tracking_data(self, tracking_data: pd.DataFrame):
        """Set enhanced tracking data for visualization."""
        self.tracking_data = tracking_data

        # Update color scheme options based on available columns
        self._update_enhanced_color_options()

        # Generate enhanced track colors
        self._generate_enhanced_track_colors()

        # Update all visualizations
        self._update_tracking_display()
        self._update_feature_analysis_options()

        self.logger.info(f"Set enhanced tracking data with {len(tracking_data)} points")

    def _update_enhanced_color_options(self):
        """Update color scheme options based on enhanced data columns."""
        if self.tracking_data is None:
            return

        # Get current selection
        current_selection = self.color_by_combo.currentText()

        # Clear and repopulate
        self.color_by_combo.clear()

        # Add standard options
        standard_options = [
            "Track ID", "Mobility Classification", "Linear Classification"
        ]
        self.color_by_combo.addItems(standard_options)

        # Add available enhanced features
        enhanced_features = [
            'scaled_rg', 'velocity', 'eigenvalue_ratio', 'step_alignment',
            'directionality_ratio', 'diffusion_coefficient', 'distanceFromOrigin',
            'radius_gyration', 'asymmetry', 'fracDimension'
        ]

        # Add density features
        density_features = [col for col in self.tracking_data.columns
                          if 'nnCountInFrame_within' in col]

        for feature in enhanced_features + density_features:
            if feature in self.tracking_data.columns:
                self.color_by_combo.addItem(feature)

        # Restore selection if possible
        index = self.color_by_combo.findText(current_selection)
        if index >= 0:
            self.color_by_combo.setCurrentIndex(index)

    def _update_feature_analysis_options(self):
        """Update feature analysis combo boxes with available features."""
        if self.tracking_data is None:
            return

        # Get numeric columns for feature analysis
        numeric_columns = []
        for col in self.tracking_data.columns:
            if self.tracking_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                numeric_columns.append(col)

        # Update combo boxes
        current_x = self.x_feature_combo.currentText()
        current_y = self.y_feature_combo.currentText()
        current_color = self.color_feature_combo.currentText()

        self.x_feature_combo.clear()
        self.y_feature_combo.clear()
        self.color_feature_combo.clear()

        self.x_feature_combo.addItems(numeric_columns)
        self.y_feature_combo.addItems(numeric_columns)
        self.color_feature_combo.addItems(["None"] + numeric_columns)

        # Set defaults
        if 'scaled_rg' in numeric_columns:
            self.x_feature_combo.setCurrentText('scaled_rg')
        if 'velocity' in numeric_columns:
            self.y_feature_combo.setCurrentText('velocity')
        if 'mobility_classification' in self.tracking_data.columns:
            self.color_feature_combo.setCurrentText('mobility_classification')

    def _generate_enhanced_track_colors(self):
        """Generate enhanced colors based on classifications."""
        if self.tracking_data is None:
            return

        # Clear existing colors
        self.track_colors = {}

        if 'track_number' not in self.tracking_data.columns:
            return

        unique_tracks = self.tracking_data['track_number'].dropna().unique()

        # Color by classification if available
        if 'mobility_classification' in self.tracking_data.columns:
            track_mobility = self.tracking_data.groupby('track_number')['mobility_classification'].first()

            for track_id in unique_tracks:
                mobility = track_mobility.get(track_id, 'unclassified')
                self.track_colors[track_id] = self.classification_colors.get(
                    mobility, self.classification_colors['unclassified']
                )
        else:
            # Default color generation
            colormap = self.colormaps.get(self.current_colormap)
            n_tracks = len(unique_tracks)

            for i, track_id in enumerate(unique_tracks):
                if colormap is not None and n_tracks > 0:
                    color_val = i / max(1, n_tracks - 1)
                    try:
                        rgba = colormap.mapToQColor(color_val)
                        self.track_colors[track_id] = (rgba.red(), rgba.green(), rgba.blue(), rgba.alpha())
                    except Exception:
                        # Fallback
                        np.random.seed(int(track_id))
                        color = tuple(np.random.randint(50, 255, 3)) + (255,)
                        self.track_colors[track_id] = color

    def _update_tracking_display(self):
        """Update enhanced tracking data display."""
        if self.tracking_data is None:
            return

        # Clear existing items
        for item in self.track_items + self.overlay_items:
            if item in self.image_plot.items:
                self.image_plot.removeItem(item)
        self.track_items.clear()
        self.overlay_items.clear()

        # Apply enhanced filtering
        display_data = self._apply_enhanced_filters()

        if len(display_data) == 0:
            return

        # Display localizations with enhanced coloring
        if self.show_localizations:
            self._display_enhanced_localizations(display_data)

        # Display tracks with enhanced visualization
        if self.show_tracks and 'track_number' in display_data.columns:
            self._display_enhanced_tracks(display_data)

        # Display overlays
        if self.show_mobility_overlay:
            self._display_mobility_overlay(display_data)

        if self.show_linearity_overlay:
            self._display_linearity_overlay(display_data)

    def _apply_enhanced_filters(self) -> pd.DataFrame:
        """Apply enhanced filtering options."""
        if self.tracking_data is None:
            return pd.DataFrame()

        # Start with frame filtering
        if 'frame' in self.tracking_data.columns:
            if self.show_tracks:
                frame_range = range(
                    max(0, self.current_frame - self.track_length_limit),
                    self.current_frame + 1
                )
                display_data = self.tracking_data[
                    self.tracking_data['frame'].isin(frame_range)
                ]
            else:
                display_data = self.tracking_data[
                    self.tracking_data['frame'] == self.current_frame
                ]
        else:
            display_data = self.tracking_data.copy()

        # Apply mobility filter
        mobility_filter = self.mobility_filter_combo.currentText()
        if mobility_filter != "All" and 'mobility_classification' in display_data.columns:
            if mobility_filter == "Mobile":
                display_data = display_data[display_data['mobility_classification'] == 'mobile']
            elif mobility_filter == "Immobile":
                display_data = display_data[display_data['mobility_classification'] == 'immobile']

        # Apply linearity filter
        linearity_filter = self.linearity_filter_combo.currentText()
        if linearity_filter != "All" and 'linear_classification' in display_data.columns:
            filter_map = {
                "Linear Unidirectional": 'linear_unidirectional',
                "Linear Bidirectional": 'linear_bidirectional',
                "Non-linear": 'non_linear'
            }
            if linearity_filter in filter_map:
                display_data = display_data[
                    display_data['linear_classification'] == filter_map[linearity_filter]
                ]

        # Apply track length filter
        min_length = self.min_track_length_spin.value()
        if 'track_number' in display_data.columns:
            track_lengths = display_data.groupby('track_number').size()
            valid_tracks = track_lengths[track_lengths >= min_length].index
            display_data = display_data[display_data['track_number'].isin(valid_tracks)]

        return display_data

    def _display_enhanced_localizations(self, data: pd.DataFrame):
        """Display localizations with enhanced coloring."""
        if 'x' not in data.columns or 'y' not in data.columns:
            return

        x_orig, y_orig = data['x'].values, data['y'].values

        # Apply coordinate transformation
        if self.image_data is not None:
            x, y = y_orig, x_orig  # Simple swap for now
        else:
            x, y = x_orig, y_orig

        # Get enhanced colors
        colors = self._get_enhanced_point_colors(data)

        # Create scatter plot with enhanced styling
        self.scatter_item.setData(
            x=x, y=y,
            size=self.point_size,
            pen=pg.mkPen(None),
            brush=colors,
            symbol='o'
        )

    def _get_enhanced_point_colors(self, data: pd.DataFrame):
        """Get enhanced colors for points based on current scheme."""
        color_scheme = self.color_by_combo.currentText()

        if color_scheme == "Mobility Classification" and 'mobility_classification' in data.columns:
            colors = []
            for classification in data['mobility_classification']:
                colors.append(self.classification_colors.get(
                    classification, self.classification_colors['unclassified']
                ))
            return colors

        elif color_scheme == "Linear Classification" and 'linear_classification' in data.columns:
            colors = []
            for classification in data['linear_classification']:
                colors.append(self.classification_colors.get(
                    classification, self.classification_colors['unclassified']
                ))
            return colors

        elif color_scheme in data.columns:
            # Color by numeric feature
            values = data[color_scheme].fillna(0).values
            if len(values) > 0 and values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
                colormap = self.colormaps.get(self.current_colormap)
                if colormap is not None:
                    try:
                        return [colormap.mapToQColor(val) for val in normalized]
                    except Exception:
                        pass

        # Default: color by track ID
        if 'track_number' in data.columns:
            colors = []
            for track_id in data['track_number']:
                if pd.isna(track_id):
                    colors.append(self.classification_colors['unclassified'])
                else:
                    colors.append(self.track_colors.get(
                        track_id, self.classification_colors['unclassified']
                    ))
            return colors

        return [(255, 255, 255, 255)] * len(data)

    def _display_enhanced_tracks(self, data: pd.DataFrame):
        """Display tracks with enhanced visualization."""
        if 'track_number' not in data.columns:
            return

        tracks = data.groupby('track_number')

        for track_id, track_data in tracks:
            if len(track_data) < 2:
                continue

            track_data = track_data.sort_values('frame')
            x_orig, y_orig = track_data['x'].values, track_data['y'].values

            # Apply coordinate transformation
            if self.image_data is not None:
                x, y = y_orig, x_orig
            else:
                x, y = x_orig, y_orig

            # Get enhanced track style
            color, width, style = self._get_enhanced_track_style(track_id, track_data)

            # Create line plot
            line_item = pg.PlotDataItem(
                x=x, y=y,
                pen=pg.mkPen(color, width=width, style=style),
                connect='all'
            )

            self.image_plot.addItem(line_item)
            self.track_items.append(line_item)

            # Add enhanced track annotations
            if self.show_track_ids:
                self._add_track_annotation(track_id, x[-1], y[-1], track_data)

    def _get_enhanced_track_style(self, track_id: int, track_data: pd.DataFrame) -> Tuple:
        """Get enhanced track styling based on classifications."""
        # Default style
        color = self.track_colors.get(track_id, self.classification_colors['unclassified'])
        width = 2
        style = Qt.PenStyle.SolidLine

        # Modify based on mobility classification
        if 'mobility_classification' in track_data.columns:
            mobility = track_data['mobility_classification'].iloc[0]
            if mobility == 'mobile':
                width = 3
            elif mobility == 'immobile':
                width = 1

        # Modify based on linearity classification
        if 'linear_classification' in track_data.columns:
            linearity = track_data['linear_classification'].iloc[0]
            if linearity == 'linear_unidirectional':
                style = Qt.PenStyle.SolidLine
            elif linearity == 'linear_bidirectional':
                style = Qt.PenStyle.DashLine
            elif linearity == 'non_linear':
                style = Qt.PenStyle.DotLine

        return color, width, style

    def _add_track_annotation(self, track_id: int, x: float, y: float, track_data: pd.DataFrame):
        """Add enhanced track annotation with classification info."""
        # Build annotation text
        annotation_parts = [str(track_id)]

        if 'mobility_classification' in track_data.columns:
            mobility = track_data['mobility_classification'].iloc[0]
            annotation_parts.append(f"M:{mobility[0].upper()}")

        if 'linear_classification' in track_data.columns:
            linearity = track_data['linear_classification'].iloc[0]
            annotation_parts.append(f"L:{linearity[0].upper()}")

        annotation_text = " ".join(annotation_parts)

        # Create text item
        text_item = pg.TextItem(
            annotation_text,
            color=self.track_colors.get(track_id, (255, 255, 255)),
            anchor=(0, 0)
        )
        text_item.setPos(x, y)
        self.image_plot.addItem(text_item)
        self.track_items.append(text_item)

    def _display_mobility_overlay(self, data: pd.DataFrame):
        """Display mobility classification overlay."""
        if 'mobility_classification' not in data.columns:
            return

        # Create overlay regions for different mobility classes
        # This could be implemented as colored regions or symbols
        pass

    def _display_linearity_overlay(self, data: pd.DataFrame):
        """Display linearity classification overlay."""
        if 'linear_classification' not in data.columns:
            return

        # Create overlay showing linearity characteristics
        # This could show principal axes or directionality arrows
        pass

    def _update_density_heatmap(self):
        """Update density heatmap visualization."""
        if self.tracking_data is None:
            return

        try:
            # Get current frame data
            if 'frame' in self.tracking_data.columns:
                frame_data = self.tracking_data[
                    self.tracking_data['frame'] == self.current_frame
                ]
            else:
                frame_data = self.tracking_data

            if len(frame_data) < 2:
                self.density_image_item.clear()
                return

            # Create density heatmap
            positions = frame_data[['x', 'y']].values

            # Determine bounds
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

            # Create grid
            x_bins = np.linspace(x_min, x_max, self.heatmap_resolution)
            y_bins = np.linspace(y_min, y_max, self.heatmap_resolution)

            # Calculate density at each grid point
            density_map = np.zeros((self.heatmap_resolution, self.heatmap_resolution))

            for i, x_center in enumerate(x_bins):
                for j, y_center in enumerate(y_bins):
                    # Count particles within radius
                    distances = np.sqrt(
                        (positions[:, 0] - x_center)**2 +
                        (positions[:, 1] - y_center)**2
                    )
                    density_map[j, i] = np.sum(distances <= self.density_radius)

            # Update density image
            self.density_image_item.setImage(density_map,
                                           pos=[x_min, y_min],
                                           scale=[(x_max-x_min)/self.heatmap_resolution,
                                                 (y_max-y_min)/self.heatmap_resolution])

            # Apply colormap
            colormap = self.colormaps.get(self.current_colormap)
            if colormap:
                self.density_image_item.setColorMap(colormap)

        except Exception as e:
            self.logger.error(f"Error updating density heatmap: {e}")

    def _update_feature_plot(self):
        """Update feature analysis scatter plot."""
        if self.tracking_data is None:
            return

        x_feature = self.x_feature_combo.currentText()
        y_feature = self.y_feature_combo.currentText()
        color_feature = self.color_feature_combo.currentText()

        if not x_feature or not y_feature:
            return

        try:
            # Clear existing plots
            self.feature_graphics.clear()

            # Get track-level data
            if 'track_number' in self.tracking_data.columns:
                track_data = self.tracking_data.groupby('track_number').first()
            else:
                track_data = self.tracking_data

            if x_feature not in track_data.columns or y_feature not in track_data.columns:
                return

            x_values = track_data[x_feature].dropna()
            y_values = track_data[y_feature].dropna()

            # Align data
            common_index = x_values.index.intersection(y_values.index)
            x_values = x_values[common_index]
            y_values = y_values[common_index]

            if len(x_values) == 0:
                return

            # Create scatter plot
            scatter_plot = self.feature_graphics.addPlot(row=0, col=0)
            scatter_plot.setLabel('left', y_feature)
            scatter_plot.setLabel('bottom', x_feature)

            # Color by feature if specified
            if color_feature != "None" and color_feature in track_data.columns:
                color_values = track_data[color_feature][common_index].dropna()

                if len(color_values) > 0:
                    # Normalize color values
                    if color_values.dtype in ['object', 'category']:
                        # Categorical coloring
                        unique_vals = color_values.unique()
                        colors = []
                        for val in color_values:
                            color_idx = list(unique_vals).index(val) / max(1, len(unique_vals) - 1)
                            colors.append(color_idx)
                    else:
                        # Numeric coloring
                        colors = (color_values - color_values.min()) / (color_values.max() - color_values.min())
                else:
                    colors = 'white'
            else:
                colors = 'white'

            # Create scatter plot
            scatter = pg.ScatterPlotItem(
                x=x_values.values,
                y=y_values.values,
                brush=colors if isinstance(colors, str) else [pg.intColor(int(c*255)) for c in colors],
                size=8,
                pen=pg.mkPen(None)
            )

            scatter_plot.addItem(scatter)

        except Exception as e:
            self.logger.error(f"Error updating feature plot: {e}")

    # Event handlers
    def _toggle_tracks(self, checked: bool):
        """Toggle track display."""
        self.show_tracks = checked
        self._update_tracking_display()

    def _toggle_localizations(self, checked: bool):
        """Toggle localization points display."""
        self.show_localizations = checked
        self._update_tracking_display()

    def _toggle_track_ids(self, checked: bool):
        """Toggle track ID labels."""
        self.show_track_ids = checked
        self._update_tracking_display()

    def _toggle_mobility_overlay(self, checked: bool):
        """Toggle mobility overlay."""
        self.show_mobility_overlay = checked
        self._update_tracking_display()

    def _toggle_linearity_overlay(self, checked: bool):
        """Toggle linearity overlay."""
        self.show_linearity_overlay = checked
        self._update_tracking_display()

    def _change_point_size(self, size: int):
        """Change point size."""
        self.point_size = size
        self._update_tracking_display()

    def _change_colormap(self, colormap_name: str):
        """Change colormap."""
        if colormap_name in self.colormaps:
            self.current_colormap = colormap_name
            self._update_image_display()
            self._generate_enhanced_track_colors()
            self._update_tracking_display()

    def _change_color_scheme(self, scheme: str):
        """Change color scheme for enhanced visualization."""
        self.color_by_feature = scheme
        self._update_tracking_display()

    def _filter_by_mobility(self, mobility: str):
        """Filter by mobility classification."""
        self._update_tracking_display()

    def _filter_by_linearity(self, linearity: str):
        """Filter by linearity classification."""
        self._update_tracking_display()

    def _filter_by_length(self, min_length: int):
        """Filter by minimum track length."""
        self._update_tracking_display()

    def _update_density_radius(self, radius: int):
        """Update density analysis radius."""
        self.density_radius = radius

    def _update_density_resolution(self, resolution: int):
        """Update density heatmap resolution."""
        self.heatmap_resolution = resolution

    def _show_classification_legend(self):
        """Show classification color legend."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel

        dialog = QDialog(self)
        dialog.setWindowTitle("Classification Legend")
        dialog.setFixedSize(300, 400)

        layout = QVBoxLayout(dialog)

        # Mobility classifications
        layout.addWidget(QLabel("<b>Mobility Classifications:</b>"))
        for classification, color in self.classification_colors.items():
            if classification in ['mobile', 'immobile', 'unclassified']:
                color_label = QLabel("■")
                color_label.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]}); font-size: 20px;")
                text_label = QLabel(classification.replace('_', ' ').title())

                row_layout = QHBoxLayout()
                row_layout.addWidget(color_label)
                row_layout.addWidget(text_label)
                row_layout.addStretch()

                layout.addLayout(row_layout)

        layout.addWidget(QLabel(""))
        layout.addWidget(QLabel("<b>Linearity Classifications:</b>"))

        linearity_classifications = ['linear_unidirectional', 'linear_bidirectional', 'non_linear']
        for classification in linearity_classifications:
            if classification in self.classification_colors:
                color = self.classification_colors[classification]
                color_label = QLabel("■")
                color_label.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]}); font-size: 20px;")
                text_label = QLabel(classification.replace('_', ' ').title())

                row_layout = QHBoxLayout()
                row_layout.addWidget(color_label)
                row_layout.addWidget(text_label)
                row_layout.addStretch()

                layout.addLayout(row_layout)

        dialog.exec()

    def export_current_view(self):
        """Export current visualization."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )

        if file_path:
            try:
                current_tab = self.viz_tabs.currentIndex()
                if current_tab == 0:  # Main visualization
                    exporter = pg.exporters.ImageExporter(self.image_plot)
                elif current_tab == 1:  # Density heatmap
                    exporter = pg.exporters.ImageExporter(self.density_plot)
                elif current_tab == 2:  # Feature analysis
                    exporter = pg.exporters.ImageExporter(self.feature_graphics.scene())

                exporter.export(file_path)
                self.logger.info(f"Exported visualization to {file_path}")

            except Exception as e:
                self.logger.error(f"Error exporting visualization: {e}")

    # Public interface methods (inherited and enhanced)
    def set_image_data(self, image_data: np.ndarray):
        """Set image data with enhanced processing."""
        self.image_data = image_data
        self.current_frame = 0
        self._update_image_display()
        self.zoom_fit()
        self.logger.info(f"Set enhanced image data with shape: {image_data.shape}")

    def set_frame(self, frame: int):
        """Set current frame with enhanced updates."""
        if self.image_data is None:
            return

        max_frames = 1 if len(self.image_data.shape) == 2 else self.image_data.shape[0]
        frame = max(0, min(frame, max_frames - 1))

        if frame != self.current_frame:
            self.current_frame = frame
            self._update_image_display()
            self._update_tracking_display()
            self._update_density_heatmap()
            self.frameChanged.emit(frame)

    def _update_image_display(self):
        """Update image display with enhanced processing."""
        if self.image_data is None:
            return

        if len(self.image_data.shape) == 2:
            image = self.image_data
        elif len(self.image_data.shape) == 3:
            image = self.image_data[self.current_frame]
        else:
            return

        colormap = self.colormaps.get(self.current_colormap)
        if colormap is not None:
            try:
                self.image_item.setImage(image, levels=(image.min(), image.max()))
                self.image_item.setColorMap(colormap)
            except Exception as e:
                self.logger.warning(f"Error applying colormap: {e}")
                self.image_item.setImage(image, levels=(image.min(), image.max()))
        else:
            self.image_item.setImage(image, levels=(image.min(), image.max()))

    def _on_mouse_click(self, event):
        """Enhanced mouse click handling."""
        if self.image_view.sceneBoundingRect().contains(event.scenePos()):
            pos = self.image_view.mapSceneToView(event.scenePos())

            click_x_display = pos.x()
            click_y_display = pos.y()

            # Reverse coordinate transformation
            if self.image_data is not None:
                click_x_orig = click_y_display
                click_y_orig = click_x_display
            else:
                click_x_orig = click_x_display
                click_y_orig = click_y_display

            self.pointClicked.emit(click_x_orig, click_y_orig)

            if self.tracking_data is not None:
                self._check_enhanced_track_selection(click_x_orig, click_y_orig)

    def _check_enhanced_track_selection(self, x: float, y: float, tolerance: float = 5.0):
        """Enhanced track selection with detailed information."""
        if 'x' not in self.tracking_data.columns or 'y' not in self.tracking_data.columns:
            return

        distances = np.sqrt(
            (self.tracking_data['x'] - x)**2 + (self.tracking_data['y'] - y)**2
        )

        min_distance = distances.min()
        if min_distance <= tolerance:
            nearest_idx = distances.idxmin()
            if 'track_number' in self.tracking_data.columns:
                track_id = self.tracking_data.loc[nearest_idx, 'track_number']
                if not pd.isna(track_id):
                    self.trackSelected.emit(int(track_id))
                    self._show_track_details(int(track_id))

    def _show_track_details(self, track_id: int):
        """Show detailed track information."""
        if self.tracking_data is None:
            return

        track_data = self.tracking_data[self.tracking_data['track_number'] == track_id]
        if len(track_data) == 0:
            return

        # Build detailed information
        details = [f"Track {track_id} Details:", "=" * 20]

        # Basic info
        details.append(f"Length: {len(track_data)} points")
        if 'frame' in track_data.columns:
            details.append(f"Duration: {track_data['frame'].max() - track_data['frame'].min() + 1} frames")

        # Enhanced features
        enhanced_features = [
            ('mobility_classification', 'Mobility'),
            ('linear_classification', 'Linearity'),
            ('scaled_rg', 'Scaled Rg'),
            ('velocity', 'Mean Velocity'),
            ('eigenvalue_ratio', 'Eigenvalue Ratio'),
            ('diffusion_coefficient', 'Diffusion Coeff')
        ]

        for col, label in enhanced_features:
            if col in track_data.columns:
                value = track_data[col].iloc[0]
                if pd.notna(value):
                    if isinstance(value, (int, float)):
                        details.append(f"{label}: {value:.3f}")
                    else:
                        details.append(f"{label}: {value}")

        # Show details in log or status
        self.logger.info("\n".join(details))

    def _on_data_loaded(self, data_name: str, data: Any):
        """Enhanced data loading handler."""
        if isinstance(data, np.ndarray) and len(data.shape) >= 2:
            self.set_image_data(data)
        elif isinstance(data, pd.DataFrame):
            if 'x' in data.columns and 'y' in data.columns:
                self.set_tracking_data(data)

    # Additional utility methods
    def zoom_fit(self):
        """Zoom to fit all data."""
        self.image_view.autoRange()

    def reset_view(self):
        """Reset view to default zoom level."""
        if self.image_data is not None:
            shape = self.image_data.shape[-2:]
            self.image_view.setRange(
                xRange=[0, shape[1]],
                yRange=[0, shape[0]],
                padding=0.1
            )

    def highlight_track(self, track_id: int):
        """Highlight specific track with enhanced styling."""
        if self.tracking_data is None or 'track_number' not in self.tracking_data.columns:
            return

        track_data = self.tracking_data[self.tracking_data['track_number'] == track_id]

        if len(track_data) > 0:
            x_orig, y_orig = track_data['x'].values, track_data['y'].values

            if self.image_data is not None:
                x, y = y_orig, x_orig
            else:
                x, y = x_orig, y_orig

            # Create highlighted track with enhanced styling
            highlight_item = pg.PlotDataItem(
                x=x, y=y,
                pen=pg.mkPen((255, 255, 0), width=6),  # Thick yellow highlight
                connect='all'
            )

            self.image_plot.addItem(highlight_item)
            self.track_items.append(highlight_item)

    def clear_overlays(self):
        """Clear all overlay items."""
        for item in self.track_items + self.overlay_items:
            if item in self.image_plot.items:
                self.image_plot.removeItem(item)
        self.track_items.clear()
        self.overlay_items.clear()
        self.scatter_item.clear()
