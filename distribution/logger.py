# logger.py
import logging
import logging.handlers
import os
import threading
from datetime import datetime
import sys

class ThreadSafeLogger:
    """线程安全日志记录器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ThreadSafeLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._initialized = True
            self.loggers = {}
            self.main_logger = None
            self._setup_main_logger()
    
    def _setup_main_logger(self):
        """设置主日志记录器"""
        # 创建logs目录
        os.makedirs('logs', exist_ok=True)
        
        # 主日志文件
        log_filename = f"logs/simulation_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 配置根日志记录器
        self.main_logger = logging.getLogger('main')
        self.main_logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.main_logger.handlers:
            # 文件handler - 按文件大小轮转
            file_handler = logging.handlers.RotatingFileHandler(
                log_filename, 
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            
            # 控制台handler
            console_handler = logging.StreamHandler(sys.stdout)
            
            # 日志格式
            formatter = logging.Formatter(
                '%(asctime)s | %(threadName)-12s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.main_logger.addHandler(file_handler)
            self.main_logger.addHandler(console_handler)
        
        self.main_logger.propagate = False
    
    def get_case_logger(self, case_name, log_dir=None):
        """获取特定case的日志记录器"""
        if case_name in self.loggers:
            return self.loggers[case_name]
        
        if log_dir is None:
            log_dir = 'logs'
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建case特定的日志记录器
        logger = logging.getLogger(f'case.{case_name}')
        logger.setLevel(logging.INFO)
        
        # case日志文件
        case_log_file = os.path.join(log_dir, f"{case_name}.log")
        file_handler = logging.FileHandler(case_log_file, encoding='utf-8')
        
        # 简化格式，只记录消息
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.propagate = False  # 避免传播到根日志记录器
        
        self.loggers[case_name] = logger
        return logger
    
    def info(self, message, case_name=None):
        """记录info级别日志"""
        if case_name and case_name in self.loggers:
            self.loggers[case_name].info(message)
        else:
            self.main_logger.info(message)
    
    def error(self, message, case_name=None):
        """记录error级别日志"""
        if case_name and case_name in self.loggers:
            self.loggers[case_name].error(message)
        else:
            self.main_logger.error(message)
    
    def warning(self, message, case_name=None):
        """记录warning级别日志"""
        if case_name and case_name in self.loggers:
            self.loggers[case_name].warning(message)
        else:
            self.main_logger.warning(message)
    
    def debug(self, message, case_name=None):
        """记录debug级别日志"""
        if case_name and case_name in self.loggers:
            self.loggers[case_name].debug(message)
        else:
            self.main_logger.debug(message)
    
    def cleanup(self):
        """清理所有日志处理器"""
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        
        if self.main_logger:
            for handler in self.main_logger.handlers[:]:
                handler.close()
                self.main_logger.removeHandler(handler)

# 全局日志实例
logger = ThreadSafeLogger()

# 便捷函数
def log_info(message, case_name=None):
    logger.info(message, case_name)

def log_error(message, case_name=None):
    logger.error(message, case_name)

def log_warning(message, case_name=None):
    logger.warning(message, case_name)

def log_debug(message, case_name=None):
    logger.debug(message, case_name)