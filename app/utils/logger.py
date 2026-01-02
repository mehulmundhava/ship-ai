"""
Logging Configuration

Provides structured logging setup for the application with file rotation and zipping.
Logs are stored in the logs/ folder and automatically rotated every 24 hours.
Rotated logs are automatically zipped to save disk space.
"""

import logging
import logging.handlers
import sys
import zipfile
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class ZipRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Rotating file handler that automatically zips rotated log files.
    Rotates every 24 hours and compresses old logs.
    """
    
    def __init__(
        self,
        filename,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8',
        delay=False,
        utc=False,
        atTime=None
    ):
        super().__init__(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc,
            atTime=atTime
        )
        self.backupCount = backupCount
    
    def doRollover(self):
        """
        Override to zip the rotated file before deletion.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Get the current time for the zip filename
        current_time = datetime.now()
        time_str = current_time.strftime('%Y-%m-%d')
        
        # Find the rotated file (baseName + suffix)
        if self.backupCount > 0:
            # The TimedRotatingFileHandler creates files with suffix like .2024-01-15
            # We need to find the file that was just rotated
            base_path = Path(self.baseFilename)
            log_dir = base_path.parent
            base_name = base_path.stem
            
            # Look for the rotated file with today's date
            rotated_file = log_dir / f"{base_name}.{time_str}"
            
            if rotated_file.exists():
                # Create zip file
                zip_filename = f"{self.baseFilename}.{time_str}.zip"
                try:
                    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(str(rotated_file), rotated_file.name)
                    
                    # Remove the original rotated file
                    os.remove(str(rotated_file))
                except Exception as e:
                    # If zipping fails, log it but don't crash
                    print(f"Warning: Failed to zip log file {rotated_file}: {e}")
        
        # Perform the normal rollover
        super().doRollover()
        
        # Clean up old zip files
        self._cleanup_old_zips()
    
    def _cleanup_old_zips(self):
        """Remove zip files older than backupCount days."""
        if self.backupCount <= 0:
            return
        
        log_dir = Path(self.baseFilename).parent
        base_name = Path(self.baseFilename).stem
        
        # Find all zip files for this log
        zip_files = sorted(log_dir.glob(f"{base_name}.*.zip"), reverse=True)
        
        # Keep only the most recent backupCount files
        for old_zip in zip_files[self.backupCount:]:
            try:
                old_zip.unlink()
            except OSError:
                pass


def setup_logger(
    name: str = "ship_rag_ai",
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    rotation_interval_hours: int = 24,
    retention_days: int = 30
) -> logging.Logger:
    """
    Set up comprehensive logger with file rotation and zipping.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files (default: logs/)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        rotation_interval_hours: Hours between rotations (default: 24, uses 'midnight' when=24)
        retention_days: Days to keep zipped logs (default: 30)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Determine log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Formatter with detailed information
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handlers with rotation and zipping
    if log_to_file:
        # Main log file (all levels)
        main_log_file = log_dir / "app.log"
        file_handler = ZipRotatingFileHandler(
            filename=str(main_log_file),
            when='midnight',
            interval=1,  # Daily
            backupCount=retention_days,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Error log file (errors only)
        error_log_file = log_dir / "error.log"
        error_handler = ZipRotatingFileHandler(
            filename=str(error_log_file),
            when='midnight',
            interval=1,
            backupCount=retention_days,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    return logger


# Create default logger instance (will be reconfigured in main.py with settings)
# This is a fallback logger that will be replaced when main.py initializes
logger = setup_logger()

