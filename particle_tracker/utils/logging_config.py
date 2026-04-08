#!/usr/bin/env python3
"""
Enhanced Logging Configuration Module
====================================

Provides comprehensive logging functionality for the particle tracking application,
including performance monitoring, analysis progress tracking, and specialized
logging for different components.
"""

import os
import json
import logging
import logging.handlers
import time
import traceback
import functools
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
import threading

from PyQt6.QtCore import QObject, pyqtSignal


# ============================================================================
# LOGGING CATEGORIES AND LEVELS
# ============================================================================

class LogCategory:
    """Logging categories for different components."""
    CORE = "particle_tracker.core"
    ANALYSIS = "particle_tracker.analysis"
    DETECTION = "particle_tracker.analysis.detection"
    LINKING = "particle_tracker.analysis.linking"
    FEATURES = "particle_tracker.analysis.features"
    CLASSIFICATION = "particle_tracker.analysis.classification"
    AUTOCORR = "particle_tracker.analysis.autocorrelation"
    DENSITY = "particle_tracker.analysis.density"
    BACKGROUND = "particle_tracker.analysis.background"
    GUI = "particle_tracker.gui"
    VISUALIZATION = "particle_tracker.gui.visualization"
    DATA = "particle_tracker.data"
    BATCH = "particle_tracker.batch"
    PERFORMANCE = "particle_tracker.performance"
    MEMORY = "particle_tracker.memory"
    IO = "particle_tracker.io"


@dataclass
class LogConfig:
    """Configuration for logging setup."""
    debug: bool = False
    log_file: Optional[str] = None
    max_file_size: int = 20 * 1024 * 1024  # 20 MB
    backup_count: int = 10
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    enable_performance_logging: bool = True
    enable_memory_logging: bool = True
    enable_progress_logging: bool = True
    log_directory: Optional[str] = None
    session_logging: bool = True
    analysis_logging: bool = True


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceLogger:
    """Logger for performance monitoring and timing."""
    
    def __init__(self):
        self.logger = logging.getLogger(LogCategory.PERFORMANCE)
        self._timers = {}
        self._lock = threading.Lock()
    
    def start_timer(self, operation: str, track_id: Optional[str] = None) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{track_id}" if track_id else operation
        
        with self._lock:
            self._timers[timer_id] = {
                'start_time': time.time(),
                'operation': operation,
                'track_id': track_id
            }
        
        self.logger.debug(f"Started timing: {operation}" + 
                         (f" (track: {track_id})" if track_id else ""))
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End timing an operation and log the duration."""
        with self._lock:
            if timer_id not in self._timers:
                self.logger.warning(f"Timer not found: {timer_id}")
                return 0.0
            
            timer_info = self._timers.pop(timer_id)
        
        duration = time.time() - timer_info['start_time']
        operation = timer_info['operation']
        track_id = timer_info.get('track_id')
        
        self.logger.info(f"{operation} completed in {duration:.3f}s" + 
                        (f" (track: {track_id})" if track_id else ""))
        
        return duration
    
    @contextmanager
    def timer(self, operation: str, track_id: Optional[str] = None):
        """Context manager for timing operations."""
        timer_id = self.start_timer(operation, track_id)
        try:
            yield
        finally:
            self.end_timer(timer_id)


def performance_timer(operation_name: str = None):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            perf_logger = PerformanceLogger()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with perf_logger.timer(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# MEMORY MONITORING
# ============================================================================

class MemoryLogger:
    """Logger for memory usage monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(LogCategory.MEMORY)
        self._baseline_memory = None
        
        # Try to import psutil for memory monitoring
        try:
            import psutil
            self.psutil = psutil
            self._psutil_available = True
        except ImportError:
            self.psutil = None
            self._psutil_available = False
            self.logger.warning("psutil not available - memory monitoring will be limited")
    
    def log_memory_usage(self, label: str, track_id: Optional[str] = None):
        """Log current memory usage."""
        if not self._psutil_available:
            return
        
        try:
            process = self.psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if self._baseline_memory is None:
                self._baseline_memory = memory_mb
            
            memory_delta = memory_mb - self._baseline_memory
            
            log_msg = f"Memory usage ({label}): {memory_mb:.1f} MB"
            if memory_delta > 0:
                log_msg += f" (+{memory_delta:.1f} MB from baseline)"
            
            if track_id:
                log_msg += f" (track: {track_id})"
            
            # Log as warning if memory usage is high
            if memory_mb > 1000:  # 1 GB
                self.logger.warning(log_msg)
            else:
                self.logger.debug(log_msg)
                
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {e}")
    
    def set_baseline(self):
        """Set current memory usage as baseline."""
        if not self._psutil_available:
            return
        
        try:
            process = self.psutil.Process()
            self._baseline_memory = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory baseline set: {self._baseline_memory:.1f} MB")
        except Exception as e:
            self.logger.error(f"Error setting memory baseline: {e}")


# ============================================================================
# ANALYSIS PROGRESS LOGGING
# ============================================================================

class AnalysisProgressLogger:
    """Logger for tracking analysis progress."""
    
    def __init__(self):
        self.logger = logging.getLogger(LogCategory.ANALYSIS)
        self._current_analysis = None
        self._analysis_start_time = None
        
    def start_analysis(self, analysis_type: str, total_items: int, 
                      description: str = ""):
        """Start tracking an analysis."""
        self._current_analysis = {
            'type': analysis_type,
            'total_items': total_items,
            'completed_items': 0,
            'description': description,
            'errors': []
        }
        self._analysis_start_time = time.time()
        
        self.logger.info(f"Starting {analysis_type}: {total_items} items"
                        + (f" - {description}" if description else ""))
    
    def update_progress(self, completed_items: int, current_item: str = ""):
        """Update analysis progress."""
        if not self._current_analysis:
            return
        
        self._current_analysis['completed_items'] = completed_items
        total = self._current_analysis['total_items']
        percentage = (completed_items / total) * 100 if total > 0 else 0
        
        elapsed = time.time() - self._analysis_start_time
        items_per_sec = completed_items / elapsed if elapsed > 0 else 0
        
        log_msg = (f"Progress: {completed_items}/{total} ({percentage:.1f}%) "
                  f"- {items_per_sec:.1f} items/sec")
        
        if current_item:
            log_msg += f" - {current_item}"
        
        # Log progress at intervals
        if completed_items % max(1, total // 20) == 0:  # Every 5%
            self.logger.info(log_msg)
        else:
            self.logger.debug(log_msg)
    
    def log_error(self, error: str, item: str = ""):
        """Log an error during analysis."""
        if self._current_analysis:
            self._current_analysis['errors'].append({
                'error': error,
                'item': item,
                'timestamp': datetime.now().isoformat()
            })
        
        error_msg = f"Analysis error: {error}"
        if item:
            error_msg += f" (item: {item})"
        
        self.logger.error(error_msg)
    
    def finish_analysis(self):
        """Finish the current analysis and log summary."""
        if not self._current_analysis:
            return
        
        elapsed = time.time() - self._analysis_start_time
        analysis = self._current_analysis
        
        success_rate = ((analysis['completed_items'] - len(analysis['errors'])) / 
                       analysis['completed_items'] * 100 if analysis['completed_items'] > 0 else 0)
        
        summary = (f"Analysis complete: {analysis['type']} - "
                  f"{analysis['completed_items']}/{analysis['total_items']} items "
                  f"in {elapsed:.1f}s - {success_rate:.1f}% success rate")
        
        if analysis['errors']:
            summary += f" - {len(analysis['errors'])} errors"
        
        self.logger.info(summary)
        
        # Log errors if any
        if analysis['errors']:
            self.logger.warning(f"Errors encountered during {analysis['type']}:")
            for error in analysis['errors'][:10]:  # Limit to first 10 errors
                self.logger.warning(f"  - {error['item']}: {error['error']}")
            
            if len(analysis['errors']) > 10:
                self.logger.warning(f"  ... and {len(analysis['errors']) - 10} more errors")
        
        self._current_analysis = None
        self._analysis_start_time = None


# ============================================================================
# SESSION LOGGING
# ============================================================================

class SessionLogger:
    """Logger for tracking analysis sessions."""
    
    def __init__(self):
        self.logger = logging.getLogger(LogCategory.CORE)
        self.session_id = None
        self.session_start_time = None
        self.session_data = {}
    
    def start_session(self, session_type: str = "analysis"):
        """Start a new logging session."""
        self.session_id = f"{session_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_start_time = time.time()
        self.session_data = {
            'session_id': self.session_id,
            'session_type': session_type,
            'start_time': datetime.now().isoformat(),
            'parameters': {},
            'files_processed': [],
            'errors': [],
            'results': {}
        }
        
        self.logger.info(f"Started session: {self.session_id}")
        return self.session_id
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """Log analysis parameters for the session."""
        self.session_data['parameters'] = parameters
        self.logger.info(f"Session parameters: {json.dumps(parameters, indent=2)}")
    
    def log_file_processed(self, file_path: str, status: str, duration: float = None):
        """Log a processed file."""
        file_info = {
            'file_path': file_path,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        if duration is not None:
            file_info['duration'] = duration
        
        self.session_data['files_processed'].append(file_info)
        
        if status == 'success':
            self.logger.info(f"Processed file: {file_path}"
                           + (f" in {duration:.1f}s" if duration else ""))
        else:
            self.logger.error(f"Failed to process file: {file_path}")
    
    def log_results(self, results: Dict[str, Any]):
        """Log analysis results for the session."""
        self.session_data['results'].update(results)
        self.logger.info(f"Session results updated: {list(results.keys())}")
    
    def end_session(self, export_path: Optional[str] = None):
        """End the current session and optionally export log."""
        if not self.session_id:
            return
        
        elapsed = time.time() - self.session_start_time
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['duration'] = elapsed
        
        # Summary statistics
        total_files = len(self.session_data['files_processed'])
        successful_files = sum(1 for f in self.session_data['files_processed'] 
                              if f['status'] == 'success')
        
        self.logger.info(f"Session ended: {self.session_id} - "
                        f"Duration: {elapsed:.1f}s - "
                        f"Files: {successful_files}/{total_files} successful")
        
        # Export session log if requested
        if export_path:
            try:
                with open(export_path, 'w') as f:
                    json.dump(self.session_data, f, indent=2)
                self.logger.info(f"Session log exported to: {export_path}")
            except Exception as e:
                self.logger.error(f"Failed to export session log: {e}")
        
        self.session_id = None
        self.session_start_time = None


# ============================================================================
# ENHANCED LOGGING SETUP
# ============================================================================

def setup_enhanced_logging(config: LogConfig = None) -> Dict[str, logging.Logger]:
    """Setup enhanced logging configuration for the particle tracking application.
    
    Args:
        config: Logging configuration
        
    Returns:
        Dictionary of configured loggers by category
    """
    if config is None:
        config = LogConfig()
    
    # Create logs directory
    if config.log_directory:
        log_dir = Path(config.log_directory)
    else:
        log_dir = Path.home() / ".particle_tracker" / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    performance_formatter = logging.Formatter(
        '%(asctime)s | PERF | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_level = getattr(logging, config.console_level.upper(), logging.INFO)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    
    # Main log file handler
    if config.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"particle_tracker_{timestamp}.log"
    else:
        log_file = Path(config.log_file)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count
    )
    file_level = getattr(logging, config.file_level.upper(), logging.DEBUG)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Performance log file handler
    if config.enable_performance_logging:
        perf_file = log_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file,
            maxBytes=config.max_file_size // 4,  # Smaller performance log
            backupCount=config.backup_count
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(performance_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Configure category-specific loggers
    loggers = {}
    
    # Core application logger
    app_logger = logging.getLogger(LogCategory.CORE)
    loggers['core'] = app_logger
    
    # Analysis loggers
    for category in [LogCategory.ANALYSIS, LogCategory.DETECTION, LogCategory.LINKING,
                     LogCategory.FEATURES, LogCategory.CLASSIFICATION, LogCategory.AUTOCORR,
                     LogCategory.DENSITY, LogCategory.BACKGROUND]:
        logger = logging.getLogger(category)
        loggers[category.split('.')[-1]] = logger
    
    # GUI loggers
    gui_logger = logging.getLogger(LogCategory.GUI)
    viz_logger = logging.getLogger(LogCategory.VISUALIZATION)
    loggers['gui'] = gui_logger
    loggers['visualization'] = viz_logger
    
    # Performance logger
    if config.enable_performance_logging:
        perf_logger = logging.getLogger(LogCategory.PERFORMANCE)
        perf_logger.addHandler(perf_handler)
        loggers['performance'] = perf_logger
    
    # Memory logger
    if config.enable_memory_logging:
        memory_logger = logging.getLogger(LogCategory.MEMORY)
        loggers['memory'] = memory_logger
    
    # Data and batch loggers
    loggers['data'] = logging.getLogger(LogCategory.DATA)
    loggers['batch'] = logging.getLogger(LogCategory.BATCH)
    loggers['io'] = logging.getLogger(LogCategory.IO)
    
    # Log initialization
    app_logger.info("Enhanced logging system initialized")
    app_logger.info(f"Log directory: {log_dir}")
    app_logger.info(f"Main log file: {log_file}")
    
    if config.enable_performance_logging:
        app_logger.info(f"Performance log file: {perf_file}")
    
    return loggers


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_logger(category: str = LogCategory.CORE) -> logging.Logger:
    """Get a logger for a specific category."""
    return logging.getLogger(category)


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator to log function calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            logger.log(level, f"Calling {func_name}")
            
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"Completed {func_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}")
                logger.debug(traceback.format_exc())
                raise
        
        return wrapper
    return decorator


def log_exceptions(logger: logging.Logger):
    """Decorator to log exceptions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_name = f"{func.__module__}.{func.__name__}"
                logger.error(f"Exception in {func_name}: {e}")
                logger.debug(traceback.format_exc())
                raise
        
        return wrapper
    return decorator


# ============================================================================
# GLOBAL LOGGER INSTANCES
# ============================================================================

# These will be initialized by setup_enhanced_logging()
_global_loggers = {}
_performance_logger = None
_memory_logger = None
_progress_logger = None
_session_logger = None


def initialize_global_loggers(config: LogConfig = None):
    """Initialize global logger instances."""
    global _global_loggers, _performance_logger, _memory_logger, _progress_logger, _session_logger
    
    _global_loggers = setup_enhanced_logging(config)
    _performance_logger = PerformanceLogger()
    _memory_logger = MemoryLogger()
    _progress_logger = AnalysisProgressLogger()
    _session_logger = SessionLogger()


def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger."""
    if _performance_logger is None:
        initialize_global_loggers()
    return _performance_logger


def get_memory_logger() -> MemoryLogger:
    """Get the global memory logger."""
    if _memory_logger is None:
        initialize_global_loggers()
    return _memory_logger


def get_progress_logger() -> AnalysisProgressLogger:
    """Get the global progress logger."""
    if _progress_logger is None:
        initialize_global_loggers()
    return _progress_logger


def get_session_logger() -> SessionLogger:
    """Get the global session logger."""
    if _session_logger is None:
        initialize_global_loggers()
    return _session_logger


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Legacy compatibility function for the original setup_logging."""
    config = LogConfig(
        debug=debug,
        log_file=log_file
    )
    
    loggers = setup_enhanced_logging(config)
    return loggers['core']