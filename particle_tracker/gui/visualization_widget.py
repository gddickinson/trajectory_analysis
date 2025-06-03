#!/usr/bin/env python3
"""
Visualization Widget Module
===========================

Provides visualization capabilities for microscopy images and tracking data
using pyqtgraph for high-performance rendering.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QCheckBox, QComboBox, QGroupBox,
    QGridLayout, QColorDialog, QSpinBox
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


class VisualizationWidget(QWidget):
    """Main visualization widget for displaying images and tracking data."""

    # Signals
    frameChanged = pyqtSignal(int)
    pointClicked = pyqtSignal(float, float)  # x, y coordinates
    trackSelected = pyqtSignal(int)  # track_number

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
        self.track_length_limit = 50
        self.point_size = 3

        # Color schemes - load with error handling
        self.colormaps = self._load_colormaps()
        self.current_colormap = list(self.colormaps.keys())[0] if self.colormaps else 'gray'

        # Track colors
        self.track_colors = {}
        self.color_by_feature = None

        self._setup_ui()
        self._connect_signals()

        self.logger.info("Visualization widget initialized")

    def _load_colormaps(self) -> Dict[str, Any]:
        """Load available colormaps with error handling."""
        colormaps = {}

        # List of colormap names to try
        colormap_names = ['viridis', 'plasma', 'inferno', 'magma', 'gray', 'hot']

        for name in colormap_names:
            try:
                # Try different ways to get colormaps in pyqtgraph
                if hasattr(pg.colormap, 'get'):
                    colormap = pg.colormap.get(name)
                elif hasattr(pg, 'ColorMap'):
                    # Create basic colormaps manually if get() doesn't work
                    colormap = self._create_fallback_colormap(name)
                else:
                    colormap = self._create_fallback_colormap(name)

                if colormap is not None:
                    colormaps[name] = colormap
                    self.logger.debug(f"Loaded colormap: {name}")

            except Exception as e:
                self.logger.debug(f"Could not load built-in colormap '{name}': {e}")
                # Create a fallback colormap
                colormap = self._create_fallback_colormap(name)
                if colormap is not None:
                    colormaps[name] = colormap
                    self.logger.debug(f"Created fallback colormap: {name}")

        # Ensure we have at least one colormap
        if not colormaps:
            self.logger.warning("No colormaps could be loaded, creating basic grayscale")
            colormaps['gray'] = self._create_grayscale_colormap()

        return colormaps

    def _create_fallback_colormap(self, name: str):
        """Create a fallback colormap when the built-in one fails."""
        try:
            if name == 'gray':
                return self._create_grayscale_colormap()
            elif name == 'viridis':
                return self._create_viridis_colormap()
            elif name == 'hot':
                return self._create_hot_colormap()
            elif name == 'plasma':
                return self._create_plasma_colormap()
            elif name == 'inferno':
                return self._create_inferno_colormap()
            elif name == 'magma':
                return self._create_magma_colormap()
            else:
                # Default to grayscale for unknown colormaps
                return self._create_grayscale_colormap()
        except Exception as e:
            self.logger.error(f"Error creating fallback colormap for {name}: {e}")
            return self._create_grayscale_colormap()

    def _create_grayscale_colormap(self):
        """Create a simple grayscale colormap."""
        try:
            # Create grayscale colormap
            positions = np.linspace(0, 1, 256)
            colors = np.array([[i, i, i, 255] for i in range(256)], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception as e:
            self.logger.error(f"Error creating grayscale colormap: {e}")
            return None

    def _create_viridis_colormap(self):
        """Create a viridis-like colormap."""
        try:
            # Simplified viridis colors
            positions = np.linspace(0, 1, 5)
            colors = np.array([
                [68, 1, 84, 255],      # dark purple
                [59, 82, 139, 255],    # blue
                [33, 145, 140, 255],   # teal
                [94, 201, 98, 255],    # green
                [253, 231, 37, 255]    # yellow
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception as e:
            self.logger.error(f"Error creating viridis colormap: {e}")
            return self._create_grayscale_colormap()

    def _create_hot_colormap(self):
        """Create a hot colormap."""
        try:
            positions = np.linspace(0, 1, 4)
            colors = np.array([
                [0, 0, 0, 255],        # black
                [255, 0, 0, 255],      # red
                [255, 255, 0, 255],    # yellow
                [255, 255, 255, 255]   # white
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception as e:
            self.logger.error(f"Error creating hot colormap: {e}")
            return self._create_grayscale_colormap()

    def _create_plasma_colormap(self):
        """Create a plasma-like colormap."""
        try:
            positions = np.linspace(0, 1, 5)
            colors = np.array([
                [13, 8, 135, 255],     # dark blue
                [84, 2, 163, 255],     # purple
                [139, 10, 165, 255],   # magenta
                [185, 50, 137, 255],   # pink
                [240, 249, 33, 255]    # yellow
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception as e:
            self.logger.error(f"Error creating plasma colormap: {e}")
            return self._create_grayscale_colormap()

    def _create_inferno_colormap(self):
        """Create an inferno-like colormap."""
        try:
            positions = np.linspace(0, 1, 5)
            colors = np.array([
                [0, 0, 4, 255],        # almost black
                [40, 11, 84, 255],     # dark purple
                [101, 21, 110, 255],   # purple
                [159, 42, 99, 255],    # red-purple
                [252, 255, 164, 255]   # bright yellow
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception as e:
            self.logger.error(f"Error creating inferno colormap: {e}")
            return self._create_grayscale_colormap()

    def _create_magma_colormap(self):
        """Create a magma-like colormap."""
        try:
            positions = np.linspace(0, 1, 5)
            colors = np.array([
                [0, 0, 4, 255],        # almost black
                [28, 16, 68, 255],     # dark purple
                [79, 18, 123, 255],    # purple
                [129, 37, 129, 255],   # magenta
                [252, 253, 191, 255]   # light yellow
            ], dtype=np.uint8)
            return pg.ColorMap(pos=positions, color=colors)
        except Exception as e:
            self.logger.error(f"Error creating magma colormap: {e}")
            return self._create_grayscale_colormap()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Create graphics layout - use PlotWidget instead of ViewBox for labels
        self.graphics_widget = GraphicsLayoutWidget()
        layout.addWidget(self.graphics_widget)

        # Create plot item (not ViewBox) so we can use setLabel
        self.image_plot = self.graphics_widget.addPlot(row=0, col=0)
        self.image_plot.setAspectLocked(True)

        # Get the ViewBox from the plot item
        self.image_view = self.image_plot.getViewBox()

        # Create image item
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)

        # Create overlay for points and tracks
        self.scatter_item = pg.ScatterPlotItem()
        self.image_plot.addItem(self.scatter_item)

        # Track lines
        self.track_items = []

        # Create controls panel
        controls_panel = self._create_controls_panel()
        layout.addWidget(controls_panel)

        # Setup initial view
        self._setup_initial_view()

    def _create_controls_panel(self) -> QWidget:
        """Create the visualization controls panel."""
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

        # Point size control
        display_layout.addWidget(QLabel("Point Size:"), 2, 0)
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(self.point_size)
        self.point_size_spin.valueChanged.connect(self._change_point_size)
        display_layout.addWidget(self.point_size_spin, 2, 1)

        layout.addWidget(display_group)

        # Color options group
        color_group = QGroupBox("Color Options")
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
            "Track ID", "Frame", "Velocity", "Feature Value", "Classification"
        ])
        self.color_by_combo.currentTextChanged.connect(self._change_color_scheme)
        color_layout.addWidget(self.color_by_combo, 1, 1)

        layout.addWidget(color_group)

        # Track options group
        track_group = QGroupBox("Track Options")
        track_layout = QGridLayout(track_group)

        track_layout.addWidget(QLabel("Track Length:"), 0, 0)
        self.track_length_spin = QSpinBox()
        self.track_length_spin.setRange(1, 1000)
        self.track_length_spin.setValue(self.track_length_limit)
        self.track_length_spin.valueChanged.connect(self._change_track_length)
        track_layout.addWidget(self.track_length_spin, 0, 1)

        # Filter by classification
        track_layout.addWidget(QLabel("Show Class:"), 1, 0)
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItems(["All", "Mobile", "Immobile", "Linear", "Non-linear"])
        self.class_filter_combo.currentTextChanged.connect(self._filter_by_class)
        track_layout.addWidget(self.class_filter_combo, 1, 1)

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

        layout.addWidget(view_group)

        layout.addStretch()

        return panel

    def _setup_initial_view(self):
        """Setup initial view settings."""
        # Set axis labels using the plot item, not the ViewBox
        self.image_plot.setLabel('left', 'Y (pixels)')
        self.image_plot.setLabel('bottom', 'X (pixels)')

        # Enable mouse interaction
        self.image_view.scene().sigMouseClicked.connect(self._on_mouse_click)

    def _connect_signals(self):
        """Connect internal signals."""
        # Connect to data manager
        self.data_manager.dataLoaded.connect(self._on_data_loaded)

    def set_image_data(self, image_data: np.ndarray):
        """Set the image data for visualization.

        Args:
            image_data: Image array (2D or 3D for time series)
        """
        self.image_data = image_data
        self.current_frame = 0

        # Update display
        self._update_image_display()

        # Auto-fit view
        self.zoom_fit()

        self.logger.info(f"Set image data with shape: {image_data.shape}")

    def set_tracking_data(self, tracking_data: pd.DataFrame):
        """Set tracking data for overlay visualization.

        Args:
            tracking_data: DataFrame with tracking results
        """
        self.tracking_data = tracking_data

        # Update color scheme options based on available columns
        self._update_color_options()

        # Generate track colors
        self._generate_track_colors()

        # Update display
        self._update_tracking_display()

        self.logger.info(f"Set tracking data with {len(tracking_data)} points")

    def set_frame(self, frame: int):
        """Set the current frame for display.

        Args:
            frame: Frame number to display
        """
        if self.image_data is None:
            return

        max_frames = 1 if len(self.image_data.shape) == 2 else self.image_data.shape[0]
        frame = max(0, min(frame, max_frames - 1))

        if frame != self.current_frame:
            self.current_frame = frame
            self._update_image_display()
            self._update_tracking_display()
            self.frameChanged.emit(frame)

    def _update_image_display(self):
        """Update the image display for current frame."""
        if self.image_data is None:
            return

        if len(self.image_data.shape) == 2:
            # Single image
            image = self.image_data
        elif len(self.image_data.shape) == 3:
            # Time series
            image = self.image_data[self.current_frame]
        else:
            self.logger.warning(f"Unsupported image shape: {self.image_data.shape}")
            return

        # Apply colormap
        colormap = self.colormaps.get(self.current_colormap)
        if colormap is not None:
            try:
                self.image_item.setImage(image, levels=(image.min(), image.max()))
                self.image_item.setColorMap(colormap)
            except Exception as e:
                self.logger.warning(f"Error applying colormap: {e}")
                # Fallback: just set the image without colormap
                self.image_item.setImage(image, levels=(image.min(), image.max()))
        else:
            # Fallback: just set the image without colormap
            self.image_item.setImage(image, levels=(image.min(), image.max()))

    def _update_tracking_display(self):
        """Update the tracking data display for current frame."""
        if self.tracking_data is None:
            return

        # Clear existing track items
        for item in self.track_items:
            self.image_plot.removeItem(item)
        self.track_items.clear()

        # Filter data for current frame and nearby frames
        if 'frame' in self.tracking_data.columns:
            if self.show_tracks:
                # Show tracks with history
                frame_range = range(
                    max(0, self.current_frame - self.track_length_limit),
                    self.current_frame + 1
                )
                display_data = self.tracking_data[
                    self.tracking_data['frame'].isin(frame_range)
                ]
            else:
                # Show only current frame
                display_data = self.tracking_data[
                    self.tracking_data['frame'] == self.current_frame
                ]
        else:
            display_data = self.tracking_data

        # Apply class filter
        class_filter = self.class_filter_combo.currentText()
        if class_filter != "All" and 'mobility_classification' in display_data.columns:
            if class_filter == "Mobile":
                display_data = display_data[display_data['mobility_classification'] == 'mobile']
            elif class_filter == "Immobile":
                display_data = display_data[display_data['mobility_classification'] == 'immobile']

        # Display localizations
        if self.show_localizations and len(display_data) > 0:
            self._display_localizations(display_data)

        # Display tracks
        if self.show_tracks and 'track_number' in display_data.columns:
            self._display_tracks(display_data)

    def _display_localizations(self, data: pd.DataFrame):
        """Display localization points."""
        if 'x' not in data.columns or 'y' not in data.columns:
            return

        x_orig = data['x'].values
        y_orig = data['y'].values

        # Fix coordinate system alignment with image
        # Apply 90-degree counter-clockwise rotation + additional up-down flip
        if self.image_data is not None:
            if len(self.image_data.shape) == 2:
                image_height, image_width = self.image_data.shape
            elif len(self.image_data.shape) == 3:
                image_height, image_width = self.image_data.shape[1], self.image_data.shape[2]
            else:
                image_height, image_width = 100, 100  # fallback

            # 90-degree counter-clockwise rotation + coordinate system alignment + additional flip
            # x_new = y_orig (swap coordinates)
            # y_new = x_orig (swap coordinates, then flip)
            x = y_orig
            y = x_orig
        else:
            x, y = x_orig, y_orig

        # Determine colors
        colors = self._get_point_colors(data)

        # Create scatter plot
        self.scatter_item.setData(
            x=x, y=y,
            size=self.point_size,
            pen=pg.mkPen(None),
            brush=colors,
            symbol='o'
        )

    def _display_tracks(self, data: pd.DataFrame):
        """Display track lines."""
        if 'track_number' not in data.columns:
            return

        # Group by track
        tracks = data.groupby('track_number')

        for track_id, track_data in tracks:
            if len(track_data) < 2:
                continue

            track_data = track_data.sort_values('frame')
            x_orig = track_data['x'].values
            y_orig = track_data['y'].values

            # Fix coordinate system alignment with image
            # Apply 90-degree counter-clockwise rotation + additional up-down flip
            if self.image_data is not None:
                if len(self.image_data.shape) == 2:
                    image_height, image_width = self.image_data.shape
                elif len(self.image_data.shape) == 3:
                    image_height, image_width = self.image_data.shape[1], self.image_data.shape[2]
                else:
                    image_height, image_width = 100, 100  # fallback

                # 90-degree counter-clockwise rotation + coordinate system alignment + additional flip
                # x_new = y_orig (swap coordinates)
                # y_new = x_orig (swap coordinates, then flip)
                x = y_orig
                y = x_orig
            else:
                x, y = x_orig, y_orig

            # Get track color
            color = self._get_track_color(track_id, track_data)

            # Create line plot
            line_item = pg.PlotDataItem(
                x=x, y=y,
                pen=pg.mkPen(color, width=2),
                connect='all'
            )

            self.image_plot.addItem(line_item)
            self.track_items.append(line_item)

            # Add track ID label if enabled
            if self.show_track_ids and len(x) > 0:
                text_item = pg.TextItem(
                    str(track_id),
                    color=color,
                    anchor=(0, 0)
                )
                text_item.setPos(x[-1], y[-1])
                self.image_plot.addItem(text_item)
                self.track_items.append(text_item)

    def _get_point_colors(self, data: pd.DataFrame):
        """Get colors for localization points based on current color scheme."""
        color_scheme = self.color_by_combo.currentText()

        if color_scheme == "Track ID" and 'track_number' in data.columns:
            # Color by track ID
            colors = []
            for track_id in data['track_number']:
                if pd.isna(track_id):
                    colors.append((128, 128, 128, 255))  # Gray for unlinked
                else:
                    colors.append(self._get_track_color(track_id))
            return colors

        elif color_scheme == "Frame" and 'frame' in data.columns:
            # Color by frame number
            frames = data['frame'].values
            if len(frames) > 0 and frames.max() > frames.min():
                normalized = (frames - frames.min()) / (frames.max() - frames.min())
                colormap = self.colormaps.get(self.current_colormap)
                if colormap is not None:
                    try:
                        return [colormap.mapToQColor(val) for val in normalized]
                    except:
                        pass

        elif color_scheme == "Velocity" and 'velocity' in data.columns:
            # Color by velocity
            velocities = data['velocity'].fillna(0).values
            if len(velocities) > 0 and velocities.max() > velocities.min():
                normalized = (velocities - velocities.min()) / (velocities.max() - velocities.min())
                colormap = self.colormaps.get(self.current_colormap)
                if colormap is not None:
                    try:
                        return [colormap.mapToQColor(val) for val in normalized]
                    except:
                        pass

        elif color_scheme == "Classification":
            # Color by classification
            if 'mobility_classification' in data.columns:
                colors = []
                for classification in data['mobility_classification']:
                    if classification == 'mobile':
                        colors.append((255, 0, 0, 255))  # Red
                    elif classification == 'immobile':
                        colors.append((0, 0, 255, 255))  # Blue
                    else:
                        colors.append((128, 128, 128, 255))  # Gray
                return colors

        # Default color
        return [(255, 255, 255, 255)] * len(data)

    def _get_track_color(self, track_id: int, track_data: Optional[pd.DataFrame] = None):
        """Get color for a specific track."""
        if track_id not in self.track_colors:
            # Generate new color
            np.random.seed(int(track_id))  # Consistent colors
            color = tuple(np.random.randint(50, 255, 3)) + (255,)
            self.track_colors[track_id] = color

        return self.track_colors[track_id]

    def _generate_track_colors(self):
        """Generate colors for all tracks."""
        if self.tracking_data is None or 'track_number' not in self.tracking_data.columns:
            return

        unique_tracks = self.tracking_data['track_number'].dropna().unique()

        # Use colormap to generate distinct colors
        colormap = self.colormaps.get(self.current_colormap)
        n_tracks = len(unique_tracks)

        if colormap is not None and n_tracks > 0:
            for i, track_id in enumerate(unique_tracks):
                color_val = i / max(1, n_tracks - 1)
                try:
                    rgba = colormap.mapToQColor(color_val)
                    self.track_colors[track_id] = (rgba.red(), rgba.green(), rgba.blue(), rgba.alpha())
                except Exception as e:
                    # Fallback to random colors
                    np.random.seed(int(track_id))
                    color = tuple(np.random.randint(50, 255, 3)) + (255,)
                    self.track_colors[track_id] = color

    def _update_color_options(self):
        """Update color scheme options based on available data columns."""
        if self.tracking_data is None:
            return

        # Get available columns for coloring
        current_items = [self.color_by_combo.itemText(i)
                        for i in range(self.color_by_combo.count())]

        # Add feature columns
        feature_columns = [col for col in self.tracking_data.columns
                          if col not in ['x', 'y', 'frame', 'track_number']]

        for col in feature_columns[:10]:  # Limit to first 10 features
            if col not in current_items:
                self.color_by_combo.addItem(col)

    # Control event handlers
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

    def _change_point_size(self, size: int):
        """Change point size."""
        self.point_size = size
        self._update_tracking_display()

    def _change_colormap(self, colormap_name: str):
        """Change colormap."""
        if colormap_name in self.colormaps:
            self.current_colormap = colormap_name
            self._update_image_display()
            self._generate_track_colors()
            self._update_tracking_display()

    def _change_color_scheme(self, scheme: str):
        """Change color scheme for points and tracks."""
        self._update_tracking_display()

    def _change_track_length(self, length: int):
        """Change track length limit."""
        self.track_length_limit = length
        self._update_tracking_display()

    def _filter_by_class(self, class_name: str):
        """Filter tracks by classification."""
        self._update_tracking_display()

    def _on_mouse_click(self, event):
        """Handle mouse click events."""
        if self.image_view.sceneBoundingRect().contains(event.scenePos()):
            pos = self.image_view.mapSceneToView(event.scenePos())

            # Transform coordinates back to detection coordinate system
            click_x_display = pos.x()
            click_y_display = pos.y()

            # Reverse the coordinate transformation
            if self.image_data is not None:
                # Reverse transformation: if x_display = y_orig and y_display = x_orig
                # Then: x_orig = y_display and y_orig = x_display
                click_x_orig = click_y_display
                click_y_orig = click_x_display
            else:
                click_x_orig = click_x_display
                click_y_orig = click_y_display

            self.pointClicked.emit(click_x_orig, click_y_orig)

            # Check if click is near a track (using original coordinates)
            if self.tracking_data is not None:
                self._check_track_selection(click_x_orig, click_y_orig)

    def _check_track_selection(self, x: float, y: float, tolerance: float = 5.0):
        """Check if click is near a track and emit selection signal."""
        if 'x' not in self.tracking_data.columns or 'y' not in self.tracking_data.columns:
            return

        # Find nearest point
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

    def _on_data_loaded(self, data_name: str, data: Any):
        """Handle new data being loaded."""
        if isinstance(data, np.ndarray) and len(data.shape) >= 2:
            # Image data
            self.set_image_data(data)
        elif isinstance(data, pd.DataFrame):
            # Trajectory or localization data
            if 'x' in data.columns and 'y' in data.columns:
                self.set_tracking_data(data)

    # Public interface methods
    def zoom_fit(self):
        """Zoom to fit all data."""
        self.image_view.autoRange()

    def reset_view(self):
        """Reset view to default zoom level."""
        if self.image_data is not None:
            shape = self.image_data.shape[-2:]  # Get height, width
            self.image_view.setRange(
                xRange=[0, shape[1]],
                yRange=[0, shape[0]],
                padding=0.1
            )

    def export_image(self, filename: str):
        """Export current view as image."""
        try:
            exporter = pg.exporters.ImageExporter(self.image_plot)
            exporter.export(filename)
            self.logger.info(f"Exported image to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting image: {e}")
            return False

    def get_view_range(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get current view range."""
        x_range, y_range = self.image_view.viewRange()
        return tuple(x_range), tuple(y_range)

    def set_view_range(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """Set view range."""
        self.image_view.setRange(xRange=x_range, yRange=y_range)

    def highlight_track(self, track_id: int):
        """Highlight a specific track."""
        if self.tracking_data is None or 'track_number' not in self.tracking_data.columns:
            return

        # Create highlighted version
        track_data = self.tracking_data[self.tracking_data['track_number'] == track_id]

        if len(track_data) > 0:
            x_orig = track_data['x'].values
            y_orig = track_data['y'].values

            # Fix coordinate system alignment with image
            # Apply same transformation as other display functions
            if self.image_data is not None:
                # Simple coordinate swap
                x = y_orig
                y = x_orig
            else:
                x, y = x_orig, y_orig

            # Create highlighted line
            highlight_item = pg.PlotDataItem(
                x=x, y=y,
                pen=pg.mkPen((255, 255, 0), width=4),  # Yellow highlight
                connect='all'
            )

            self.image_plot.addItem(highlight_item)
            self.track_items.append(highlight_item)

    def clear_overlays(self):
        """Clear all overlay items."""
        for item in self.track_items:
            self.image_plot.removeItem(item)
        self.track_items.clear()

        self.scatter_item.clear()
