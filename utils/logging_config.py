"""
Enhanced Logging Configuration for SATIM GRC System
Current Date: 2025-06-13 20:26:29 UTC
Current User: LyesHADJAR
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from typing import Optional

class GRCFormatter(logging.Formatter):
    """Custom formatter for GRC analysis logs."""
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.now(timezone.utc)
    
    def format(self, record):
        # Add elapsed time since start
        current_time = datetime.now(timezone.utc)
        elapsed = (current_time - self.start_time).total_seconds()
        
        # Color codes for different levels
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        level_color = colors.get(record.levelname, colors['RESET'])
        reset_color = colors['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Create custom format
        if hasattr(record, 'domain'):
            log_format = f"{level_color}[{timestamp}] [{record.levelname:8}] [+{elapsed:6.1f}s] [{record.domain}] {record.name}: {record.getMessage()}{reset_color}"
        else:
            log_format = f"{level_color}[{timestamp}] [{record.levelname:8}] [+{elapsed:6.1f}s] {record.name}: {record.getMessage()}{reset_color}"
        
        return log_format

class AnalysisProgressFilter(logging.Filter):
    """Filter to categorize analysis progress logs."""
    
    def filter(self, record):
        # Add analysis stage context
        if hasattr(record, 'stage'):
            record.msg = f"[{record.stage}] {record.msg}"
        return True

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 enable_console: bool = True) -> None:
    """
    Setup comprehensive logging for GRC analysis.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/satim_grc_analysis_{timestamp}.log"
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # File formatter (no colors)
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )
    file_formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()
    file_handler.setFormatter(file_formatter)
    
    # Console handler with colors
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(GRCFormatter())
        console_handler.addFilter(AnalysisProgressFilter())
        root_logger.addHandler(console_handler)
    
    root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    _configure_module_loggers()
    
    # Log startup message
    logger = logging.getLogger("grc.system")
    logger.info("="*80)
    logger.info("🚀 SATIM GRC Analysis System - Enhanced Logging Initialized")
    logger.info(f"📅 Session Start: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"👤 User: LyesHADJAR")
    logger.info(f"📝 Log Level: {log_level.upper()}")
    logger.info(f"📄 Log File: {log_file}")
    logger.info("="*80)

def _configure_module_loggers():
    """Configure logging for specific modules."""
    
    # Vector search logging
    vector_logger = logging.getLogger("rag.vector_search")
    vector_logger.setLevel(logging.INFO)
    
    # Policy analysis logging
    policy_logger = logging.getLogger("agents.policy_comparison")
    policy_logger.setLevel(logging.INFO)
    
    # Feedback agent logging
    feedback_logger = logging.getLogger("agents.feedback")
    feedback_logger.setLevel(logging.INFO)
    
    # LLM interaction logging
    llm_logger = logging.getLogger("rag.llm")
    llm_logger.setLevel(logging.INFO)
    
    # Reduce verbose logging from external libraries
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def log_analysis_stage(stage: str, message: str, level: str = "INFO"):
    """Log analysis stage with context."""
    logger = logging.getLogger("grc.analysis")
    log_func = getattr(logger, level.lower())
    
    # Add stage context to log record
    extra = {'stage': stage}
    log_func(message, extra=extra)

def log_domain_analysis(domain: str, message: str, level: str = "INFO"):
    """Log domain-specific analysis."""
    logger = logging.getLogger("grc.domain")
    log_func = getattr(logger, level.lower())
    
    # Add domain context to log record
    extra = {'domain': domain}
    log_func(message, extra=extra)

def log_performance(operation: str, duration: float, details: dict = None):
    """Log performance metrics."""
    logger = logging.getLogger("grc.performance")
    
    details_str = ""
    if details:
        details_str = " | " + " | ".join([f"{k}: {v}" for k, v in details.items()])
    
    logger.info(f"⏱️ {operation}: {duration:.2f}s{details_str}")

def log_llm_interaction(prompt_length: int, response_length: int, model: str, duration: float):
    """Log LLM interactions."""
    logger = logging.getLogger("grc.llm")
    logger.info(f"🤖 LLM Call: {model} | Prompt: {prompt_length} chars | Response: {response_length} chars | Duration: {duration:.2f}s")