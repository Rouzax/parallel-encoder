"""Tests for the logging setup."""

from __future__ import annotations

import logging

from logger import setup_logging


def test_setup_logging_returns_logger():
    log = setup_logging(verbosity=0)
    assert isinstance(log, logging.Logger)
    assert log.name == "parallel-encoder"


def test_verbosity_zero_sets_warning():
    log = setup_logging(verbosity=0)
    assert log.level == logging.WARNING


def test_verbosity_one_sets_info():
    log = setup_logging(verbosity=1)
    assert log.level == logging.INFO


def test_verbosity_two_sets_debug():
    log = setup_logging(verbosity=2)
    assert log.level == logging.DEBUG


def test_file_handler_created(tmp_path):
    log_file = tmp_path / "encode.log"
    log = setup_logging(verbosity=2, log_file=str(log_file))
    file_handlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) >= 1
    log.debug("test message")
    # Flush handlers
    for h in file_handlers:
        h.flush()
    assert "test message" in log_file.read_text()
