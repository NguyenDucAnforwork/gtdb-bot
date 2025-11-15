# src/persona/persona_manager.py
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import json

@dataclass
class Persona:
    """Represents a chatbot persona with specific characteristics."""
    name: str
    description: str
    system_prompt: str
    keywords: List[str]
    specialties: List[str]
    tone: str
    response_style: str

class PersonaManager:
    """Manages different chatbot personas and routes queries to appropriate persona."""
    
    def __init__(self, llm):
        self.llm = llm
        self.personas = self._load_default_personas()
        self.router_chain = self._build_router_chain()
    
    def _load_default_personas(self) -> Dict[str, Persona]:
        """Load default persona configurations."""
        personas = {
            "general": Persona(
                name="General Assistant",
                description="A helpful general-purpose assistant",
                system_prompt="You are a helpful, knowledgeable assistant. Provide accurate, concise, and helpful responses to user questions.",
                keywords=["general", "help", "question", "information"],
                specialties=["general knowledge", "basic questions", "explanations"],
                tone="friendly and professional",
                response_style="clear and concise"
            ),
            "technical": Persona(
                name="Technical Expert",
                description="A technical expert specializing in programming and technology",
                system_prompt="You are a technical expert with deep knowledge in programming, software development, AI, and technology. Provide detailed technical explanations with code examples when appropriate.",
                keywords=["code", "programming", "technical", "development", "software", "AI", "machine learning", "algorithm"],
                specialties=["programming", "software architecture", "AI/ML", "system design"],
                tone="professional and detailed",
                response_style="technical with examples"
            ),
            "legal": Persona(
                name="Legal Advisor", 
                description="A legal expert specializing in Vietnamese law",
                system_prompt="You are a legal expert with extensive knowledge of Vietnamese law and regulations. Provide accurate legal information while reminding users to consult qualified legal professionals for specific cases.",
                keywords=["legal", "law", "regulation", "nghị định", "luật", "pháp luật", "legal", "court", "lawyer"],
                specialties=["Vietnamese law", "legal regulations", "legal procedures"],
                tone="authoritative but accessible",
                response_style="structured with legal references"
            ),
            "creative": Persona(
                name="Creative Assistant",
                description="A creative helper for writing and artistic endeavors",
                system_prompt="You are a creative assistant specializing in writing, storytelling, and artistic projects. Help users with creative tasks, brainstorming, and artistic guidance.",
                keywords=["creative", "writing", "story", "art", "design", "brainstorm", "idea", "creative"],
                specialties=["creative writing", "storytelling", "art guidance", "brainstorming"],
                tone="inspiring and imaginative",
                response_style="creative and engaging"
            ),
            "educational": Persona(
                name="Educational Tutor",
                description="An educational tutor for learning and academic help",
                system_prompt="You are an educational tutor who helps students learn. Break down complex topics into understandable parts, provide examples, and encourage learning.",
                keywords=["learn", "study", "education", "tutorial", "explain", "teach", "homework", "academic"],
                specialties=["teaching", "academic help", "learning guidance", "explanations"],
                tone="patient and encouraging",
                response_style="educational with step-by-step explanations"
            )
        }
        return personas
    
    def _build_router_chain(self):
        """Build the routing chain to select appropriate persona."""
        persona_descriptions = "\n".join([
            f"- {name}: {persona.description} (Keywords: {', '.join(persona.keywords[:3])})"
            for name, persona in self.personas.items()
        ])
        
        router_prompt = ChatPromptTemplate.from_template(
            """Given the user's question, determine which persona would be most appropriate to handle it.
            
Available personas:
{persona_descriptions}

User question: {question}

Respond with only the persona name (one of: {persona_names}). 
Consider the keywords and content of the question to make the best match.

Persona name:"""
        )
        
        return (
            router_prompt.partial(
                persona_descriptions=persona_descriptions,
                persona_names=", ".join(self.personas.keys())
            )
            | self.llm
            | StrOutputParser()
            | RunnableLambda(lambda x: x.strip().lower())
        )
    
    def route_question(self, question: str) -> str:
        """Route question to appropriate persona and return persona name."""
        try:
            selected_persona = self.router_chain.invoke({"question": question})
            
            # Validate the selected persona
            if selected_persona in self.personas:
                return selected_persona
            else:
                # Try to find partial matches
                for persona_name in self.personas.keys():
                    if persona_name in selected_persona:
                        return persona_name
                
                # Default to general if no match found
                print(f"Unknown persona '{selected_persona}', defaulting to 'general'")
                return "general"
                
        except Exception as e:
            print(f"Error in persona routing: {e}")
            return "general"
    
    def get_persona(self, persona_name: str) -> Persona:
        """Get persona by name."""
        return self.personas.get(persona_name, self.personas["general"])
    
    def get_system_prompt(self, persona_name: str) -> str:
        """Get system prompt for specified persona."""
        persona = self.get_persona(persona_name)
        return persona.system_prompt
    
    def add_persona(self, persona: Persona) -> None:
        """Add a new persona."""
        self.personas[persona.name.lower()] = persona
    
    def list_personas(self) -> List[str]:
        """List all available personas."""
        return list(self.personas.keys())
    
    def get_persona_info(self, persona_name: str) -> Dict[str, Any]:
        """Get detailed information about a persona."""
        persona = self.get_persona(persona_name)
        return {
            "name": persona.name,
            "description": persona.description,
            "specialties": persona.specialties,
            "tone": persona.tone,
            "response_style": persona.response_style,
            "keywords": persona.keywords
        }