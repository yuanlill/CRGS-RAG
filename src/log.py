# Copyright 2023 The Alpaca Team
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging

class DistributedLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that controls logging in distributed environments"""
    @staticmethod
    def should_log(main_process_only: bool) -> bool:
        """Determine if logging should occur based on process rank"""
        process_rank = int(os.getenv("LOCAL_RANK", 0))
        return not main_process_only or process_rank <= 0

    def log(self, level: int, message: str, *args, **kwargs):
        """Log message with distributed environment awareness"""
        main_process_only = kwargs.pop("main_process_only", True)
        if self.isEnabledFor(level) and self.should_log(main_process_only):
            processed_msg, processed_kwargs = self.process(message, kwargs)
            self.logger.log(level, processed_msg, *args, **processed_kwargs)

def get_logger(name: str, log_level: str = None) -> DistributedLoggerAdapter:
    """Get configured logger instance"""
    logger_instance = logging.getLogger(name)
    if log_level:
        logger_instance.setLevel(log_level.upper())
    return DistributedLoggerAdapter(logger_instance, {})

class LoggingDisabler:
    """Context manager and decorator to temporarily disable logging"""
    def __enter__(self):
        logging.disable(logging.CRITICAL)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logging.disable(logging.NOTSET)

    def __call__(self, func):
        """Decorator to disable logging during function execution"""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper