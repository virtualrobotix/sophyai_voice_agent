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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def connect(self):
        """Initialize connection pool."""
        current_loop = asyncio.get_running_loop()
        # Se il pool esiste ma è legato a un loop diverso, chiudilo
        if self.pool is not None and self._loop != current_loop:
            logger.warning("Event loop changed, recreating database pool")
            try:
                await self.pool.close()
            except Exception:
                pass
            self.pool = None
        
        if self.pool is None:
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )
                self._loop = current_loop
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
        """Ensure we have an active connection pool in the current event loop."""
        current_loop = asyncio.get_running_loop()
        if self.pool is None or self._loop != current_loop:
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
    
    # ==================== Users ====================

    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE username = $1", username)
            return dict(row) if row else None

    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            return dict(row) if row else None

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE email = $1 AND is_active = TRUE", email)
            return dict(row) if row else None

    async def get_all_users(self) -> List[Dict[str, Any]]:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, username, email, role, must_change_password, is_active, created_at, updated_at, last_login FROM users ORDER BY id")
            result = []
            for row in rows:
                u = dict(row)
                for k in ('created_at', 'updated_at', 'last_login'):
                    if u.get(k):
                        u[k] = u[k].isoformat()
                result.append(u)
            return result

    async def create_user(self, username: str, password_hash: str, email: str = None, role: str = 'user', must_change_password: bool = True) -> int:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO users (username, password_hash, email, role, must_change_password) VALUES ($1, $2, $3, $4, $5) RETURNING id",
                username, password_hash, email, role, must_change_password
            )
            return row["id"]

    async def update_user(self, user_id: int, **kwargs) -> bool:
        await self._ensure_connected()
        allowed = {'email', 'role', 'is_active', 'must_change_password', 'password_hash', 'last_login'}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return False
        set_parts = [f"{k} = ${i+2}" for i, k in enumerate(updates.keys())]
        query = f"UPDATE users SET {', '.join(set_parts)} WHERE id = $1"
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, user_id, *updates.values())
            return "UPDATE 1" in result

    async def delete_user(self, user_id: int) -> bool:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            result = await conn.execute("UPDATE users SET is_active = FALSE WHERE id = $1", user_id)
            return "UPDATE 1" in result

    async def count_users(self) -> int:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM users")
            return row["cnt"]

    async def create_password_reset_token(self, user_id: int, token: str, expires_at) -> int:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES ($1, $2, $3) RETURNING id",
                user_id, token, expires_at
            )
            return row["id"]

    async def get_password_reset_token(self, token: str) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM password_reset_tokens WHERE token = $1 AND used = FALSE AND expires_at > CURRENT_TIMESTAMP",
                token
            )
            return dict(row) if row else None

    async def mark_reset_token_used(self, token: str) -> bool:
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            result = await conn.execute("UPDATE password_reset_tokens SET used = TRUE WHERE token = $1", token)
            return "UPDATE 1" in result

    # ==================== Call Logs ====================
    
    async def create_call_log(
        self, 
        call_id: str, 
        room_name: str,
        caller_number: str = None,
        called_number: str = None,
        caller_name: str = None,
        sip_trunk_id: str = None,
        metadata: dict = None
    ) -> int:
        """Crea un nuovo log di chiamata e restituisce l'ID."""
        await self._ensure_connected()
        import json
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO call_logs (call_id, room_name, caller_number, called_number, 
                                       caller_name, sip_trunk_id, metadata, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'active')
                ON CONFLICT (call_id) DO UPDATE SET 
                    status = 'active',
                    start_time = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                call_id, room_name, caller_number, called_number,
                caller_name, sip_trunk_id, json.dumps(metadata or {})
            )
            return row["id"]
    
    async def end_call_log(self, call_id: str, status: str = "completed") -> bool:
        """Chiude un log di chiamata calcolando la durata."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE call_logs 
                SET status = $2,
                    end_time = CURRENT_TIMESTAMP,
                    duration_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - start_time))::INTEGER,
                    updated_at = CURRENT_TIMESTAMP
                WHERE call_id = $1 AND status = 'active'
                """,
                call_id, status
            )
            return "UPDATE 1" in result
    
    async def get_call_log_by_call_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Ottiene un log di chiamata per call_id."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM call_logs WHERE call_id = $1",
                call_id
            )
            if row:
                return dict(row)
            return None
    
    async def get_call_log_by_room(self, room_name: str, status: str = "active") -> Optional[Dict[str, Any]]:
        """Ottiene un log di chiamata attivo per room_name."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM call_logs 
                WHERE room_name = $1 AND status = $2
                ORDER BY start_time DESC
                LIMIT 1
                """,
                room_name, status
            )
            if row:
                return dict(row)
            return None
    
    async def add_call_message(self, call_log_id: int, role: str, content: str) -> int:
        """Aggiunge un messaggio al log della chiamata."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO call_messages (call_log_id, role, content)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                call_log_id, role, content
            )
            return row["id"]
    
    async def get_call_messages(self, call_log_id: int) -> List[Dict[str, Any]]:
        """Ottiene tutti i messaggi di una chiamata."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, timestamp
                FROM call_messages
                WHERE call_log_id = $1
                ORDER BY timestamp ASC
                """,
                call_log_id
            )
            return [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                }
                for row in rows
            ]
    
    async def get_call_logs(
        self, 
        limit: int = 50, 
        offset: int = 0,
        status: str = None,
        from_date: datetime = None,
        to_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """Ottiene la lista dei log delle chiamate con filtri."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            query = """
                SELECT cl.*, 
                       (SELECT COUNT(*) FROM call_messages WHERE call_log_id = cl.id) as message_count
                FROM call_logs cl
                WHERE 1=1
            """
            params = []
            param_count = 0
            
            if status:
                param_count += 1
                query += f" AND status = ${param_count}"
                params.append(status)
            
            if from_date:
                param_count += 1
                query += f" AND start_time >= ${param_count}"
                params.append(from_date)
            
            if to_date:
                param_count += 1
                query += f" AND start_time <= ${param_count}"
                params.append(to_date)
            
            query += " ORDER BY start_time DESC"
            
            param_count += 1
            query += f" LIMIT ${param_count}"
            params.append(limit)
            
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(offset)
            
            rows = await conn.fetch(query, *params)
            
            result = []
            for row in rows:
                call = dict(row)
                # Converti datetime in ISO format
                if call.get('start_time'):
                    call['start_time'] = call['start_time'].isoformat()
                if call.get('end_time'):
                    call['end_time'] = call['end_time'].isoformat()
                if call.get('created_at'):
                    call['created_at'] = call['created_at'].isoformat()
                if call.get('updated_at'):
                    call['updated_at'] = call['updated_at'].isoformat()
                result.append(call)
            
            return result
    
    async def get_call_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche sulle chiamate."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_calls,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed_calls,
                    COUNT(*) FILTER (WHERE status = 'active') as active_calls,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_calls,
                    COUNT(*) FILTER (WHERE status = 'missed') as missed_calls,
                    AVG(duration_seconds) FILTER (WHERE duration_seconds > 0) as avg_duration,
                    SUM(duration_seconds) FILTER (WHERE duration_seconds > 0) as total_duration,
                    COUNT(*) FILTER (WHERE start_time >= CURRENT_DATE) as today_calls
                FROM call_logs
                """
            )
            return {
                "total_calls": stats["total_calls"] or 0,
                "completed_calls": stats["completed_calls"] or 0,
                "active_calls": stats["active_calls"] or 0,
                "failed_calls": stats["failed_calls"] or 0,
                "missed_calls": stats["missed_calls"] or 0,
                "avg_duration_seconds": round(float(stats["avg_duration"] or 0), 1),
                "total_duration_seconds": int(stats["total_duration"] or 0),
                "today_calls": stats["today_calls"] or 0
            }


# Global database instance
_db_instance: Optional[DatabaseService] = None


async def get_db() -> DatabaseService:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseService()
    # Assicura che la connessione sia attiva nel loop corrente
    await _db_instance._ensure_connected()
    return _db_instance


async def close_db():
    """Close the global database instance."""
    global _db_instance
    if _db_instance:
        await _db_instance.disconnect()
        _db_instance = None





