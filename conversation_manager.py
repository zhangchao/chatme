"""
Conversation management module for handling chat history and context
"""
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import config

logger = logging.getLogger(__name__)

class ConversationEntry:
    """Represents a single conversation entry"""
    
    def __init__(self, entry_type: str, text: str, timestamp: float = None, 
                 image_path: str = None, metadata: Dict[str, Any] = None):
        self.type = entry_type  # "user" or "assistant"
        self.text = text
        self.timestamp = timestamp or time.time()
        self.image_path = image_path
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary"""
        return {
            "type": self.type,
            "text": self.text,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Create entry from dictionary"""
        return cls(
            entry_type=data.get("type", "user"),
            text=data.get("text", ""),
            timestamp=data.get("timestamp", time.time()),
            image_path=data.get("image_path"),
            metadata=data.get("metadata", {})
        )

class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = config.MAX_CONVERSATION_HISTORY):
        self.max_history = max_history
        self.conversation_history: List[ConversationEntry] = []
        self.current_session_id = self._generate_session_id()
        self.session_start_time = time.time()
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{int(time.time())}"
    
    def add_user_message(self, text: str, image_path: str = None, 
                        metadata: Dict[str, Any] = None) -> ConversationEntry:
        """Add user message to conversation"""
        entry = ConversationEntry(
            entry_type="user",
            text=text,
            image_path=image_path,
            metadata=metadata
        )
        
        self._add_entry(entry)
        logger.info(f"Added user message: {text[:50]}...")
        return entry
    
    def add_assistant_message(self, text: str, metadata: Dict[str, Any] = None) -> ConversationEntry:
        """Add assistant message to conversation"""
        entry = ConversationEntry(
            entry_type="assistant",
            text=text,
            metadata=metadata
        )
        
        self._add_entry(entry)
        logger.info(f"Added assistant message: {text[:50]}...")
        return entry
    
    def _add_entry(self, entry: ConversationEntry):
        """Add entry to conversation history"""
        self.conversation_history.append(entry)
        
        # Trim history if it exceeds max length
        if len(self.conversation_history) > self.max_history:
            removed_entries = self.conversation_history[:-self.max_history]
            self.conversation_history = self.conversation_history[-self.max_history:]
            
            # Clean up old image files
            for removed_entry in removed_entries:
                if removed_entry.image_path and Path(removed_entry.image_path).exists():
                    try:
                        Path(removed_entry.image_path).unlink()
                    except Exception as e:
                        logger.error(f"Error removing old image file: {e}")
    
    def get_recent_history(self, count: int = 10) -> List[ConversationEntry]:
        """Get recent conversation entries"""
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def get_conversation_context(self, max_entries: int = 5) -> str:
        """Get formatted conversation context for AI processing"""
        recent_entries = self.get_recent_history(max_entries)
        
        if not recent_entries:
            return "No previous conversation."
        
        context_lines = []
        for entry in recent_entries:
            prefix = "User" if entry.type == "user" else "Assistant"
            context_lines.append(f"{prefix}: {entry.text}")
        
        return "\n".join(context_lines)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current session"""
        user_messages = sum(1 for entry in self.conversation_history if entry.type == "user")
        assistant_messages = sum(1 for entry in self.conversation_history if entry.type == "assistant")
        session_duration = time.time() - self.session_start_time
        
        return {
            "session_id": self.current_session_id,
            "duration_seconds": session_duration,
            "total_messages": len(self.conversation_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "start_time": self.session_start_time
        }
    
    def save_conversation(self, filepath: str = None) -> bool:
        """Save conversation to file"""
        if not filepath:
            timestamp = int(time.time())
            filepath = config.LOGS_DIR / f"conversation_{timestamp}.json"
        
        try:
            conversation_data = {
                "session_id": self.current_session_id,
                "start_time": self.session_start_time,
                "entries": [entry.to_dict() for entry in self.conversation_history],
                "stats": self.get_session_stats()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversation saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, filepath: str) -> bool:
        """Load conversation from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            self.current_session_id = conversation_data.get("session_id", self._generate_session_id())
            self.session_start_time = conversation_data.get("start_time", time.time())
            
            # Load entries
            self.conversation_history = []
            for entry_data in conversation_data.get("entries", []):
                entry = ConversationEntry.from_dict(entry_data)
                self.conversation_history.append(entry)
            
            logger.info(f"Conversation loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return False
    
    def clear_conversation(self):
        """Clear current conversation"""
        # Clean up image files
        for entry in self.conversation_history:
            if entry.image_path and Path(entry.image_path).exists():
                try:
                    Path(entry.image_path).unlink()
                except Exception as e:
                    logger.error(f"Error removing image file: {e}")
        
        self.conversation_history.clear()
        self.current_session_id = self._generate_session_id()
        self.session_start_time = time.time()
        logger.info("Conversation cleared")
    
    def export_conversation_text(self) -> str:
        """Export conversation as plain text"""
        if not self.conversation_history:
            return "No conversation to export."
        
        lines = [
            f"Conversation Export - Session: {self.current_session_id}",
            f"Started: {time.ctime(self.session_start_time)}",
            f"Total Messages: {len(self.conversation_history)}",
            "=" * 50,
            ""
        ]
        
        for entry in self.conversation_history:
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            prefix = "👤 User" if entry.type == "user" else "🤖 Assistant"
            lines.append(f"[{timestamp_str}] {prefix}: {entry.text}")
            
            if entry.image_path:
                lines.append(f"    📷 Image: {entry.image_path}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def get_conversation_summary(self) -> str:
        """Get a brief summary of the conversation"""
        if not self.conversation_history:
            return "No conversation yet."
        
        stats = self.get_session_stats()
        duration_minutes = stats["duration_seconds"] / 60
        
        return (f"Session active for {duration_minutes:.1f} minutes. "
                f"{stats['user_messages']} user messages, "
                f"{stats['assistant_messages']} assistant responses.")
