# src/memory/conversation_memory.py
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Optional
import json

class EnhancedConversationMemory:
    """Enhanced conversation memory with follow-up handling capabilities."""
    
    def __init__(self, session_id: str, llm=None, max_token_limit: int = 2000):
        self.session_id = session_id
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.chat_history = ChatMessageHistory()
        
        # Buffer memory for recent messages
        self.buffer_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            chat_memory=self.chat_history,
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )
        
        # Summary memory for older conversations (if LLM provided)
        if llm:
            self.summary_memory = ConversationSummaryBufferMemory(
                llm=llm,
                memory_key="chat_history",
                chat_memory=self.chat_history,
                return_messages=True,
                max_token_limit=max_token_limit
            )
        else:
            self.summary_memory = None
    
    def add_user_message(self, message: str) -> None:
        """Add user message to memory."""
        self.chat_history.add_user_message(message)
    
    def add_ai_message(self, message: str) -> None:
        """Add AI message to memory."""  
        self.chat_history.add_ai_message(message)
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages from memory."""
        if self.summary_memory:
            return self.summary_memory.chat_memory.messages
        return self.buffer_memory.chat_memory.messages
    
    def get_context_for_followup(self) -> str:
        """Get formatted context for follow-up question handling."""
        messages = self.get_messages()
        if not messages:
            return ""
        
        # Format recent conversation for context
        context_parts = []
        for i, msg in enumerate(messages[-6:]):  # Last 3 exchanges
            if isinstance(msg, HumanMessage):
                context_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear conversation memory."""
        self.chat_history.clear()

class ConversationContextualizer:
    """Handles contextualizing follow-up questions based on conversation history."""
    
    def __init__(self, llm):
        self.llm = llm
        self.contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Given a chat history and the latest user question which might reference "
             "the chat history, formulate a standalone question which can be understood "
             "without the chat history. Do NOT answer the question, just reformulate it "
             "if needed and otherwise return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        self.contextualize_chain = (
            self.contextualize_prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def contextualize_question(self, question: str, chat_history: List[BaseMessage]) -> str:
        """
        Contextualize a question based on chat history.
        Returns a standalone question that doesn't require history context.
        """
        if not chat_history:
            return question
        
        try:
            result = self.contextualize_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            return result.strip()
        except Exception as e:
            print(f"Error contextualizing question: {e}")
            return question

def get_conversation_memory(session_id: str, llm=None):
    """Creates and returns enhanced conversation memory."""
    print(f"Creating enhanced conversation memory for session ID: {session_id}")
    return EnhancedConversationMemory(session_id, llm)

def get_contextualizer(llm):
    """Creates and returns conversation contextualizer."""
    return ConversationContextualizer(llm)
