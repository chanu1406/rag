import sys
from loguru import logger
from src.config import AppConfig

def setup_logging(config: AppConfig):
    """
    Configure the global logger settings.
    
    Args:
        config: Checked AppConfig object
    """
    # Remove default handler
    logger.remove()
    
    # Determine log level from config
    log_level = config.system.log_level.upper()
    
    # Add console handler (stderr)
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler (rotating)
    log_file = config.system.data_dir.parent / "logs" / "app.log"
    logger.add(
        log_file,
        rotation="10 MB",
        retention="1 week",
        level="DEBUG", # Always log debug to file for post-mortem
        compression="zip"
    )
    
    logger.info(f"Logging initialized at level {log_level}")
