from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Set
import asyncio
import json
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time features"""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Connections grouped by type (stt, chat, etc.)
        self.connections_by_type: Dict[str, Set[str]] = {
            "stt": set(),
            "chat": set(),
            "general": set()
        }
        
        # User sessions mapping
        self.user_sessions: Dict[str, Set[str]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, connection_type: str = "general", user_id: str = None) -> str:
        """Accept a new WebSocket connection and return connection ID"""
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[connection_id] = websocket
        
        # Add to type group
        if connection_type not in self.connections_by_type:
            self.connections_by_type[connection_type] = set()
        self.connections_by_type[connection_type].add(connection_id)
        
        # Track user session if provided
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(connection_id)
        
        # Store metadata
        self.connection_metadata[connection_id] = {
            "type": connection_type,
            "user_id": user_id,
            "connected_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        logger.info(f"New {connection_type} connection: {connection_id} (user: {user_id})")
        
        return connection_id
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        connection_id = None
        
        # Find connection ID by websocket instance
        for conn_id, ws in self.active_connections.items():
            if ws == websocket:
                connection_id = conn_id
                break
        
        if connection_id:
            self._remove_connection(connection_id)
    
    def disconnect_by_id(self, connection_id: str):
        """Remove a connection by its ID"""
        self._remove_connection(connection_id)
    
    def _remove_connection(self, connection_id: str):
        """Internal method to remove a connection"""
        if connection_id not in self.active_connections:
            return
        
        metadata = self.connection_metadata.get(connection_id, {})
        connection_type = metadata.get("type", "general")
        user_id = metadata.get("user_id")
        
        # Remove from active connections
        del self.active_connections[connection_id]
        
        # Remove from type group
        if connection_type in self.connections_by_type:
            self.connections_by_type[connection_type].discard(connection_id)
        
        # Remove from user session
        if user_id and user_id in self.user_sessions:
            self.user_sessions[user_id].discard(connection_id)
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
        
        # Remove metadata
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"Connection disconnected: {connection_id} (type: {connection_type}, user: {user_id})")
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send a message to a specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
                self._update_activity(connection_id)
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                self._remove_connection(connection_id)
    
    async def send_to_user(self, message: dict, user_id: str):
        """Send a message to all connections for a specific user"""
        if user_id in self.user_sessions:
            connections = list(self.user_sessions[user_id])
            await self._send_to_connections(message, connections)
    
    async def send_to_type(self, message: dict, connection_type: str):
        """Send a message to all connections of a specific type"""
        if connection_type in self.connections_by_type:
            connections = list(self.connections_by_type[connection_type])
            await self._send_to_connections(message, connections)
    
    async def broadcast(self, message: dict):
        """Send a message to all active connections"""
        connections = list(self.active_connections.keys())
        await self._send_to_connections(message, connections)
    
    async def _send_to_connections(self, message: dict, connection_ids: List[str]):
        """Send a message to multiple connections"""
        if not connection_ids:
            return
        
        # Add timestamp to message
        message["timestamp"] = datetime.now().isoformat()
        
        tasks = []
        for connection_id in connection_ids:
            if connection_id in self.active_connections:
                tasks.append(self.send_personal_message(message, connection_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _update_activity(self, connection_id: str):
        """Update last activity timestamp for a connection"""
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["last_activity"] = datetime.now()
    
    def get_connection_count(self, connection_type: str = None) -> int:
        """Get count of active connections"""
        if connection_type:
            return len(self.connections_by_type.get(connection_type, set()))
        return len(self.active_connections)
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user"""
        return list(self.user_sessions.get(user_id, set()))
    
    def get_connection_info(self, connection_id: str) -> dict:
        """Get metadata for a specific connection"""
        return self.connection_metadata.get(connection_id, {})
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        stats = {
            "total_connections": len(self.active_connections),
            "connections_by_type": {
                conn_type: len(connections) 
                for conn_type, connections in self.connections_by_type.items()
            },
            "active_users": len(self.user_sessions),
            "oldest_connection": None,
            "newest_connection": None
        }
        
        # Find oldest and newest connections
        if self.connection_metadata:
            connections_with_time = [
                (conn_id, meta["connected_at"]) 
                for conn_id, meta in self.connection_metadata.items()
            ]
            connections_with_time.sort(key=lambda x: x[1])
            
            stats["oldest_connection"] = {
                "id": connections_with_time[0][0],
                "connected_at": connections_with_time[0][1].isoformat()
            }
            stats["newest_connection"] = {
                "id": connections_with_time[-1][0], 
                "connected_at": connections_with_time[-1][1].isoformat()
            }
        
        return stats
    
    async def cleanup_inactive_connections(self, max_idle_minutes: int = 30):
        """Clean up connections that have been inactive for too long"""
        now = datetime.now()
        inactive_connections = []
        
        for connection_id, metadata in self.connection_metadata.items():
            last_activity = metadata.get("last_activity", metadata.get("connected_at", now))
            idle_minutes = (now - last_activity).total_seconds() / 60
            
            if idle_minutes > max_idle_minutes:
                inactive_connections.append(connection_id)
        
        for connection_id in inactive_connections:
            logger.info(f"Cleaning up inactive connection: {connection_id}")
            
            # Close the websocket
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                try:
                    await websocket.close(code=1000, reason="Connection timeout")
                except:
                    pass  # Connection might already be closed
            
            # Remove from manager
            self._remove_connection(connection_id)
        
        return len(inactive_connections)

# Global connection manager instance
connection_manager = ConnectionManager() 