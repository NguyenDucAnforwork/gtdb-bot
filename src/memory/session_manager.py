# src/memory/session_manager.py
from .conversation_memory import get_conversation_memory

class SessionManager:
    """
    A simple session manager to hold conversation memories for different users.
    """
    def __init__(self):
        self.sessions = {}

    def reset_all_sessions(self):
        """Clear all stored sessions (useful for tests or debugging)."""
        self.sessions = {}

    
    def create_session(self, session_id: str):
        """
        Creates a new session with a conversation memory.
        """
        self.sessions[session_id] = get_conversation_memory(session_id)

    def get_memory(self, session_id: str):
        """
        Retrieves the memory for a session, creating it if it doesn't exist.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = get_conversation_memory(session_id)
        return self.sessions[session_id]

# Global instance
session_manager = SessionManager()
