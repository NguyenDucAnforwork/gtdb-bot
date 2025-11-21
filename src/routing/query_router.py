# src/routing/query_router.py
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
import json
import re

@dataclass
class RouteConfig:
    """Configuration for a query route."""
    name: str
    description: str
    keywords: List[str]
    patterns: List[str]
    priority: int = 1
    handler_type: str = "default"
    
class QueryRouter:
    """Advanced query routing system with multiple routing strategies."""
    
    def __init__(self, llm):
        self.llm = llm
        self.routes = self._load_default_routes()
        self.llm_router = self._build_llm_router()
        
    def _load_default_routes(self) -> Dict[str, RouteConfig]:
        """Load default route configurations."""
        routes = {
            "web_search": RouteConfig(
                name="web_search",
                description="Questions requiring current/recent information, news, or real-time data",
                keywords=["latest", "current", "news", "recent", "today", "now", "update", "what's happening"],
                patterns=[r"what.*latest", r"current.*status", r"news.*about", r"what.*happening"],
                priority=3,
                handler_type="web_search"
            ),
            "factual_qa": RouteConfig(
                name="factual_qa", 
                description="Factual questions with definitive answers",
                keywords=["what is", "who is", "when", "where", "how many", "definition", "meaning"],
                patterns=[r"what is", r"who is", r"when was", r"where is", r"how many"],
                priority=2,
                handler_type="knowledge_base"
            ),
            "legal_query": RouteConfig(
                name="legal_query",
                description="Questions about Vietnamese law, regulations, and legal matters",
                keywords=["luật", "nghị định", "quy định", "pháp luật", "legal", "law", "regulation"],
                patterns=[r"nghị định", r"luật.*số", r"quy định.*về", r"theo.*luật"],
                priority=3,
                handler_type="legal_specialist"
            ),
            "technical_help": RouteConfig(
                name="technical_help",
                description="Programming, technical, and development questions",
                keywords=["code", "programming", "python", "javascript", "algorithm", "debug", "error"],
                patterns=[r"how to.*code", r".*programming", r".*algorithm", r"debug.*error"],
                priority=2,
                handler_type="technical_specialist"
            ),
            "conversational": RouteConfig(
                name="conversational",
                description="General conversation, greetings, and chat",
                keywords=["hello", "hi", "thanks", "how are you", "chat", "talk"],
                patterns=[r"^(hi|hello|hey)", r"how are you", r"thank you", r"thanks"],
                priority=1,
                handler_type="conversation"
            ),
            "calculation": RouteConfig(
                name="calculation",
                description="Mathematical calculations and computations",
                keywords=["calculate", "compute", "math", "arithmetic", "plus", "minus", "multiply", "divide"],
                patterns=[r"\d+.*[+\-*/].*\d+", r"calculate.*", r"what.*\d+.*\d+"],
                priority=3,
                handler_type="calculator"
            )
        }
        return routes
    
    def _build_llm_router(self):
        """Build LLM-based router for complex routing decisions."""
        route_descriptions = "\n".join([
            f"- {route.name}: {route.description}"
            for route in self.routes.values()
        ])
        
        router_prompt = ChatPromptTemplate.from_template(
            """Analyze the user's question and determine the most appropriate route for handling it.

Available routes:
{route_descriptions}

User question: {question}

Consider:
1. The type of information being requested
2. Whether it requires real-time data or can be answered from existing knowledge
3. The domain/field of the question
4. The complexity and nature of the query

Respond with only the route name from the list above.

Route:"""
        )
        
        return (
            router_prompt.partial(route_descriptions=route_descriptions)
            | self.llm 
            | StrOutputParser()
            | RunnableLambda(lambda x: x.strip().lower())
        )
    
    def route_query(self, question: str) -> Dict[str, Any]:
        """
        Route a query using multiple strategies and return routing decision.
        Returns dict with route name, confidence, and metadata.
        """
        # Strategy 1: Pattern-based routing (highest priority)
        pattern_route = self._route_by_patterns(question)
        if pattern_route:
            return {
                "route": pattern_route["route"],
                "confidence": 0.9,
                "strategy": "pattern_match",
                "metadata": pattern_route
            }
        
        # Strategy 2: Keyword-based routing
        keyword_route = self._route_by_keywords(question)
        if keyword_route:
            return {
                "route": keyword_route["route"], 
                "confidence": 0.7,
                "strategy": "keyword_match",
                "metadata": keyword_route
            }
        
        # Strategy 3: LLM-based routing (fallback)
        try:
            llm_route = self.llm_router.invoke({"question": question})
            if llm_route in self.routes:
                return {
                    "route": llm_route,
                    "confidence": 0.6,
                    "strategy": "llm_routing",
                    "metadata": {"llm_decision": llm_route}
                }
        except Exception as e:
            print(f"LLM routing failed: {e}")
        
        # Default fallback
        return {
            "route": "factual_qa",
            "confidence": 0.3,
            "strategy": "default_fallback", 
            "metadata": {"reason": "no_match_found"}
        }
    
    def _route_by_patterns(self, question: str) -> Optional[Dict[str, Any]]:
        """Route query using regex patterns."""
        question_lower = question.lower()
        
        for route_name, route_config in self.routes.items():
            for pattern in route_config.patterns:
                if re.search(pattern, question_lower):
                    return {
                        "route": route_name,
                        "matched_pattern": pattern,
                        "priority": route_config.priority
                    }
        return None
    
    def _route_by_keywords(self, question: str) -> Optional[Dict[str, Any]]:
        """Route query using keyword matching."""
        question_lower = question.lower()
        
        best_match = None
        max_score = 0
        
        for route_name, route_config in self.routes.items():
            score = 0
            matched_keywords = []
            
            for keyword in route_config.keywords:
                if keyword.lower() in question_lower:
                    score += route_config.priority
                    matched_keywords.append(keyword)
            
            if score > max_score:
                max_score = score
                best_match = {
                    "route": route_name,
                    "matched_keywords": matched_keywords,
                    "score": score,
                    "priority": route_config.priority
                }
        
        return best_match if max_score > 0 else None
    
    def add_route(self, route_config: RouteConfig) -> None:
        """Add a new route configuration."""
        self.routes[route_config.name] = route_config
    
    def get_route_config(self, route_name: str) -> Optional[RouteConfig]:
        """Get configuration for a specific route."""
        return self.routes.get(route_name)
    
    def list_routes(self) -> List[str]:
        """List all available routes."""
        return list(self.routes.keys())
    
    def get_handler_type(self, route_name: str) -> str:
        """Get handler type for a route."""
        route = self.routes.get(route_name)
        return route.handler_type if route else "default"

class RouteBranch:
    """Handles branching logic for different route types."""
    
    def __init__(self, router: QueryRouter):
        self.router = router
        
    def create_routing_chain(self, handlers: Dict[str, Any]):
        """
        Create a routing chain that branches to different handlers based on route.
        
        Args:
            handlers: Dict mapping route names to handler functions/chains
        """
        def route_and_execute(inputs):
            question = inputs["question"] 
            routing_result = self.router.route_query(question)
            route_name = routing_result["route"]
            
            # Add routing metadata to inputs
            inputs["routing_info"] = routing_result
            
            # Get appropriate handler
            handler = handlers.get(route_name, handlers.get("default"))
            
            if handler:
                return handler.invoke(inputs)
            else:
                return f"No handler found for route: {route_name}"
        
        return RunnableLambda(route_and_execute)