"""
Database module for Voice Agent.
Provides PostgreSQL-based persistence for settings, chats, and messages.
"""

from .database import DatabaseService, get_db

__all__ = ["DatabaseService", "get_db"]



