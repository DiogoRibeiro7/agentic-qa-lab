"""Environment adapters.

Environments isolate all browser I/O behind the
:class:`~agentic_qa_lab.environments.base.BrowserEnvironment` interface so the
rest of the system depends only on domain types.
"""

from __future__ import annotations

from .api_env import APIEnvironment
from .appium_env import AppiumEnvironment
from .base import BrowserEnvironment
from .playwright_env import DEFAULT_TIMEOUT_MS, PlaywrightEnvironment
from .selenium_env import SeleniumEnvironment

__all__ = [
    "AppiumEnvironment",
    "APIEnvironment",
    "BrowserEnvironment",
    "DEFAULT_TIMEOUT_MS",
    "PlaywrightEnvironment",
    "SeleniumEnvironment",
]
