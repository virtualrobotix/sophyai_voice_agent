"""
Database Service for Voice Agent.
Provides async PostgreSQL operations for settings, chats, and messages.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncpg
from loguru import logger


class DatabaseService:
    """Async database service for PostgreSQL operations."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "postgresql://voiceagent:voiceagent_pwd@localhost:5432/voiceagent"
        )
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Initialize connection pool."""
        if self.pool is None:
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )
                logger.info("Database connection pool created")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise
    
    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    async def _ensure_connected(self):
        """Ensure we have an active connection pool."""
        if self.pool is None:
            await self.connect()
    
    # ==================== Settings ====================
    
    async def get_setting(self, key: str) -> Optional[str]:
        """Get a single setting value."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM settings WHERE key = $1",
                key
            )
            return row["value"] if row else None
    
    async def set_setting(self, key: str, value: str) -> None:
        """Set a setting value (upsert)."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO settings (key, value) VALUES ($1, $2)
                ON CONFLICT (key) DO UPDATE SET value = $2
                """,
                key, value
            )
    
    async def get_all_settings(self) -> Dict[str, str]:
        """Get all settings as a dictionary."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT key, value FROM settings")
            return {row["key"]: row["value"] for row in rows}
    
    async def set_multiple_settings(self, settings: Dict[str, str]) -> None:
        """Set multiple settings at once."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for key, value in settings.items():
                    await conn.execute(
                        """
                        INSERT INTO settings (key, value) VALUES ($1, $2)
                        ON CONFLICT (key) DO UPDATE SET value = $2
                        """,
                        key, value
                    )
    
    # ==================== Chats ====================
    
    async def get_chats(self) -> List[Dict[str, Any]]:
        """Get all chats ordered by most recent."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, created_at, updated_at,
                       (SELECT COUNT(*) FROM messages WHERE chat_id = chats.id) as message_count
                FROM chats 
                ORDER BY updated_at DESC
                """
            )
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "message_count": row["message_count"]
                }
                for row in rows
            ]
    
    async def create_chat(self, title: str = "Nuova Chat") -> int:
        """Create a new chat and return its ID."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO chats (title) VALUES ($1) RETURNING id",
                title
            )
            return row["id"]
    
    async def get_chat(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """Get a single chat by ID."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, title, created_at, updated_at FROM chats WHERE id = $1",
                chat_id
            )
            if row:
                return {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat()
                }
            return None
    
    async def update_chat_title(self, chat_id: int, title: str) -> bool:
        """Update chat title."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE chats SET title = $1 WHERE id = $2",
                title, chat_id
            )
            return result == "UPDATE 1"
    
    async def delete_chat(self, chat_id: int) -> bool:
        """Delete a chat and all its messages."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chats WHERE id = $1",
                chat_id
            )
            return result == "DELETE 1"
    
    # ==================== Messages ====================
    
    async def get_messages(self, chat_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages for a chat."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, created_at 
                FROM messages 
                WHERE chat_id = $1 
                ORDER BY created_at ASC
                LIMIT $2
                """,
                chat_id, limit
            )
            return [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"].isoformat()
                }
                for row in rows
            ]
    
    async def add_message(self, chat_id: int, role: str, content: str) -> int:
        """Add a message to a chat and return its ID."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            # Add message
            row = await conn.fetchrow(
                """
                INSERT INTO messages (chat_id, role, content) 
                VALUES ($1, $2, $3) 
                RETURNING id
                """,
                chat_id, role, content
            )
            # Update chat's updated_at
            await conn.execute(
                "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                chat_id
            )
            return row["id"]
    
    async def delete_message(self, message_id: int) -> bool:
        """Delete a single message."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM messages WHERE id = $1",
                message_id
            )
            return result == "DELETE 1"


# Global database instance
_db_instance: Optional[DatabaseService] = None


async def get_db() -> DatabaseService:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseService()
        await _db_instance.connect()
    return _db_instance


async def close_db():
    """Close the global database instance."""
    global _db_instance
    if _db_instance:
        await _db_instance.disconnect()
        _db_instance = None


