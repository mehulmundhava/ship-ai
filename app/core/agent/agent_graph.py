"""
LangGraph Agent Implementation

This module implements the agentic RAG workflow using LangGraph.
The agent can conditionally call tools to retrieve examples or execute queries.
"""

import json
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.utilities.sql_database import SQLDatabase
from app.core.agent.agent_tools import (
    create_get_few_shot_examples_tool, 
    create_execute_db_query_tool,
    create_get_table_list_tool,
    create_get_table_structure_tool,
    create_count_query_tool,
    create_list_query_tool,
    create_get_extra_examples_tool,
    create_journey_list_tool,
    create_journey_count_tool
)
from app.core.prompts import get_system_prompt
from app.services.vector_store_service import VectorStoreService
from app.config.settings import settings
import logging
import re
import time

# Get logger for this module
logger = logging.getLogger("ship_rag_ai")


def _log_messages_debug(messages: List, step_name: str):
    """Log message sequence only at DEBUG level."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[{step_name}] messages={len(messages)}")


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    question: str
    user_id: Optional[str]
    messages: Annotated[List, "messages"]
    examples_retrieved: bool
    sql_query: Optional[str]
    query_result: Optional[str]
    final_answer: Optional[str]
    iteration_count: int  # Track iterations to prevent infinite loops
    token_usage: Optional[Dict[str, int]]  # Track token usage: input, output, total
    query_validated: Optional[bool]  # Track if user query has been validated
    llm_call_history: Optional[List[Dict[str, Any]]]  # Track what was sent/received for each LLM call
    stage_breakdown: Optional[List[Dict[str, Any]]]  # Per-stage timing and token counts for request summary
    csv_id: Optional[str]  # CSV download ID for API response (relative path: /download-csv/{csv_id})
    csv_download_path: Optional[str]  # Relative path e.g. /download-csv/{uuid} for UI/Postman


class SQLAgentGraph:
    """
    LangGraph-based SQL agent with RAG capabilities.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        db: SQLDatabase,
        vector_store_manager: VectorStoreService,
        user_id: Optional[str] = None,
        top_k: int = 20
    ):
        """
        Initialize the SQL agent graph.
        
        Args:
            llm: Language model instance (OpenAI, Groq, or any BaseChatModel)
            db: SQLDatabase instance
            vector_store_manager: VectorStoreService instance
            user_id: User ID for access control
            top_k: Maximum number of results
        """
        self.llm = llm
        self.db = db
        self.vector_store_manager = vector_store_manager
        self.user_id = user_id
        self.top_k = top_k
        
        # Store provider info for fallback logic
        from app.services.llm_service import LLMService
        llm_service = LLMService()
        self.provider = llm_service.get_provider()
        
        # Create tools
        self.get_examples_tool = create_get_few_shot_examples_tool(vector_store_manager)
        self.execute_db_query_tool = create_execute_db_query_tool(db, vector_store_manager)
        self.get_table_list_tool = create_get_table_list_tool(db)
        self.get_table_structure_tool = create_get_table_structure_tool(db)
        self.count_query_tool = create_count_query_tool(db)
        self.list_query_tool = create_list_query_tool(db)
        self.get_extra_examples_tool = create_get_extra_examples_tool(vector_store_manager)
        self.journey_list_tool = create_journey_list_tool(db, user_id)
        self.journey_count_tool = create_journey_count_tool(db, user_id)
        
        # Store the underlying query tool for direct execution
        from app.core.agent.agent_tools import QuerySQLDatabaseTool
        self.query_tool_instance = QuerySQLDatabaseTool(db)
        
        # OPTIMIZATION 1: Store tools separately for conditional binding
        # Journey tools (only needed for journey questions)
        self.journey_tools = [
            self.journey_list_tool,
            self.journey_count_tool,
            self.execute_db_query_tool  # Still needed for SQL generation
        ]
        
        # Regular tools (for non-journey questions)
        # OPTIMIZATION: Reduced tool set to save tokens
        # Removed: count_query_tool, list_query_tool (execute_db_query handles these)
        # Removed: get_table_list, get_table_structure (fallback handled via auto-retry)
        self.regular_tools = [
            self.execute_db_query_tool,
            self.get_examples_tool
        ]
        
        # Keep all tools for ToolNode (needs access to all tools for execution)
        # But we'll bind conditionally to LLM to save tokens
        self.tools = self.journey_tools + [
            tool for tool in self.regular_tools if tool not in self.journey_tools
        ]
        
        # Don't bind tools yet - will bind conditionally based on question type
        self.llm_with_tools = None
        self._current_tools_bound = None  # Track which tools are currently bound
        
        # Build graph
        self.graph = self._build_graph()
    
    def _is_journey_question(self, question: str) -> bool:
        """Check if question is about journeys."""
        if not question:
            return False
        question_lower = question.lower()
        journey_keywords = [
            "journey", "journeys", "movement", "facility to facility",
            "entered", "exited", "path", "traveled", "transition"
        ]
        return any(keyword in question_lower for keyword in journey_keywords)
    
    def _bind_tools_conditionally(self, question: str):
        """Bind tools based on question type to save tokens."""
        is_journey = self._is_journey_question(question)
        
        if is_journey:
            tools_to_bind = self.journey_tools
            tool_type = "journey"
        else:
            tools_to_bind = self.regular_tools
            tool_type = "regular"
        
        # Only re-bind if tools changed
        if self._current_tools_bound != tool_type:
            logger.debug(f"Binding {tool_type} tools ({len(tools_to_bind)} tools)")
            # Groq-specific: Some models need explicit tool_choice or have tool calling issues
            # Try standard bind_tools first, but we'll catch errors and retry if needed
            try:
                if self.provider == "GROQ":
                    # For Groq, ensure tools are properly formatted
                    # Some Groq models work better without explicit tool_choice
                    self.llm_with_tools = self.llm.bind_tools(tools_to_bind)
                else:
                    self.llm_with_tools = self.llm.bind_tools(tools_to_bind)
            except Exception as e:
                logger.warning(f"Error binding tools to LLM: {e}. Will retry on first invocation.")
                # Set to None so we can retry with fallback on first use
                self.llm_with_tools = None
            
            self._current_tools_bound = tool_type
    
    def _get_fallback_llm(self, use_tools=False, question=""):
        """
        Get or create fallback LLM instance (ChatGPT).
        
        Args:
            use_tools: Whether to bind tools to fallback LLM
            question: Question text to determine which tools to bind
        
        Returns:
            BaseChatModel: Configured OpenAI LLM instance
        """
        if not hasattr(self, '_fallback_llm') or self._fallback_llm is None:
            # Import here to avoid circular dependency
            from app.services.llm_service import LLMService
            llm_service = LLMService()
            self._fallback_llm = llm_service.get_fallback_llm_model()
            
            logger.debug("Created ChatGPT fallback LLM instance")
        
        # If tools are needed, bind them
        if use_tools:
            # Determine which tools to bind (same as regular LLM)
            is_journey = self._is_journey_question(question) if question else False
            tools_to_bind = self.journey_tools if is_journey else self.regular_tools
            
            # Create a unique key for this tool binding
            tool_key = f"journey" if is_journey else "regular"
            cache_key = f'_fallback_llm_with_tools_{tool_key}'
            
            if not hasattr(self, cache_key) or getattr(self, cache_key) is None:
                setattr(self, cache_key, self._fallback_llm.bind_tools(tools_to_bind))
            return getattr(self, cache_key)
        else:
            return self._fallback_llm
    
    def _invoke_groq_with_recovery(self, messages, use_tools=False, question=""):
        """
        Groq-specific invocation with recovery mechanisms.
        - If use_tools=False: Direct invocation (for Security Guard, Format Answer)
        - If use_tools=True: Tries tool calling first, then without tools, then extracts SQL from text.
        
        Args:
            messages: List of messages to send to LLM
            use_tools: Whether to use tool binding
            question: Question text for context
        
        Returns:
            LLM response or synthetic AIMessage with tool calls, or None if all recovery fails
        """
        model_name = getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model', None) or 'Unknown'
        logger.debug(f"Using GROQ LLM: {model_name}")
        
        # Simple path: No tools needed (Security Guard, Format Answer, etc.)
        if not use_tools:
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                logger.warning(f"Groq invocation failed (no tools): {error_type} - {error_str[:200]}")
                # For non-tool calls, just return None to trigger OpenAI fallback
                return None
        
        # Tool calling path: Try with recovery mechanisms
        error_str = None
        error_type = None
        
        # Step 1: Try with tool binding (if requested)
        if self.llm_with_tools:
            try:
                return self.llm_with_tools.invoke(messages)
            except Exception as tool_error:
                error_str = str(tool_error)
                error_type = type(tool_error).__name__
                if "tool_use_failed" in error_str or "Failed to call a function" in error_str:
                    logger.info("🔧 Groq tool calling failed, attempting recovery...")
                    logger.debug("Groq tool calling failed, attempting recovery")
                    # Continue to recovery steps below
                else:
                    # Non-tool-calling error, log and continue to recovery
                    logger.warning(f"Groq tool invocation error: {error_type} - {error_str[:200]}")
        
        # Step 2: Try without tool binding (Groq might generate SQL in text)
        if use_tools:
            try:
                logger.info("🔧 Attempting Groq recovery: invoking without tool binding...")
                logger.debug("Attempting Groq recovery: invoking without tool binding")
                response_without_tools = self.llm.invoke(messages)
                
                # Check if response contains SQL
                if hasattr(response_without_tools, 'content') and response_without_tools.content:
                    content = response_without_tools.content
                    # Try to extract SQL from the text response
                    sql_match = re.search(r'(?:SELECT|WITH|INSERT|UPDATE|DELETE)\s+.*?(?:;|$)', content, re.IGNORECASE | re.DOTALL)
                    if sql_match:
                        sql_query = sql_match.group(0).strip().rstrip(';')
                        logger.info(f"✅ Extracted SQL from Groq text response: {sql_query[:100]}...")
                        logger.debug("Extracted SQL from Groq text response")
                        # Create synthetic tool response
                        synthetic_response = self._create_synthetic_tool_response(
                            "execute_db_query",
                            {"query": sql_query},
                            messages
                        )
                        return synthetic_response
                    else:
                        logger.warning("Groq response doesn't contain SQL")
            except Exception as recovery_error:
                logger.warning(f"Groq recovery (no tools) failed: {recovery_error}")
                if not error_str:
                    error_str = str(recovery_error)
                    error_type = type(recovery_error).__name__
        
        # Step 3: Try to extract SQL from error message (if we have one)
        if error_str and ("tool_use_failed" in error_str or "Failed to call a function" in error_str):
            logger.info("🔧 Attempting to extract SQL from Groq error message...")
            logger.debug("Attempting to extract SQL from Groq error message")
            extracted = self._extract_sql_from_groq_error(error_str)
            if extracted and use_tools:
                try:
                    synthetic_response = self._create_synthetic_tool_response(
                        extracted.get("tool_name", "execute_db_query"),
                        extracted.get("args", {}),
                        messages
                    )
                    logger.info("✅ Created synthetic response from Groq error extraction")
                    logger.debug("Recovered from Groq error - extracted SQL")
                    return synthetic_response
                except Exception as synth_error:
                    logger.warning(f"Failed to create synthetic response: {synth_error}")
        
        # If all Groq recovery attempts fail, return None to trigger OpenAI fallback
        logger.warning("All Groq recovery mechanisms failed")
        return None
    
    def _invoke_openai(self, messages, use_tools=False, question=""):
        """
        OpenAI-specific invocation (direct, no special recovery needed).
        
        Args:
            messages: List of messages to send to LLM
            use_tools: Whether to use tool binding
            question: Question text for tool binding
        
        Returns:
            LLM response
        """
        model_name = getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model', None) or 'Unknown'
        logger.debug(f"Using OPENAI LLM: {model_name}")
        logger.debug(f"Using OPENAI LLM: {model_name}")
        
        if use_tools and self.llm_with_tools:
            return self.llm_with_tools.invoke(messages)
        else:
            return self.llm.invoke(messages)
    
    def _invoke_llm_with_fallback(self, messages, use_tools=False, question=""):
        """
        Invoke LLM with provider-specific handling and automatic fallback.
        - Groq: Uses recovery mechanisms (tool calling → no tools → SQL extraction)
        - OpenAI: Direct invocation
        - Fallback: Only if Groq completely fails, use OpenAI
        
        Args:
            messages: List of messages to send to LLM
            use_tools: Whether to use tool binding (llm_with_tools)
            question: Question text for tool binding (if use_tools=True)
        
        Returns:
            LLM response or synthetic AIMessage with tool calls
        """
        # Route to provider-specific handler
        if self.provider == "GROQ":
            # Try Groq with recovery mechanisms
            groq_response = self._invoke_groq_with_recovery(messages, use_tools, question)
            if groq_response is not None:
                return groq_response
            
            # Groq recovery failed, fall back to OpenAI
            logger.warning("Groq recovery failed, falling back to OpenAI")
        else:
            # OpenAI: direct invocation
            try:
                return self._invoke_openai(messages, use_tools, question)
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                logger.error(f"OpenAI invocation failed: {error_type} - {error_str[:200]}")
                logger.debug(f"OpenAI Error: {error_type}")
                raise
        
        # If we get here, Groq failed and we need to fall back to OpenAI
        try:
            fallback_llm = self._get_fallback_llm(use_tools, question)
            fallback_model_name = getattr(fallback_llm, 'model_name', None) or getattr(fallback_llm, 'model', None) or 'Unknown'
            logger.debug(f"Using FALLBACK LLM (OpenAI): {fallback_model_name}")
            logger.debug(f"Using FALLBACK LLM (OpenAI): {fallback_model_name}")
            
            response = fallback_llm.invoke(messages)
            logger.debug("Got response from OpenAI fallback")
            logger.debug("Got response from OpenAI fallback")
            return response
        except Exception as e2:
            logger.error(f"OpenAI fallback also failed: {e2}")
            logger.exception("OpenAI fallback error")
            raise
        
        # This should never be reached, but just in case
        raise Exception("All LLM invocation attempts failed")
    
    def _extract_sql_from_groq_error(self, error_str: str) -> Optional[Dict[str, Any]]:
        """
        Extract SQL query and tool name from Groq's tool_use_failed error.
        
        Groq sometimes generates function calls in wrong format:
        '<function=execute_db_query {"query": "SELECT ..."}></function>'
        
        Args:
            error_str: Error message string containing failed_generation
            
        Returns:
            Dict with 'tool_name' and 'args', or None if extraction fails
        """
        try:
            logger.debug(f"Attempting to extract from error string (length: {len(error_str)})")
            
            # The error format is: 'failed_generation': '<function=execute_db_query {"query": "..."}></function>'
            # We need to extract the content between the single quotes after 'failed_generation'
            
            # Method 1: Extract the entire failed_generation value (handles nested quotes)
            # Look for: 'failed_generation': '<function=...></function>'
            # We'll extract everything between the single quotes after the colon
            failed_gen_pattern = r"'failed_generation':\s*'((?:[^'\\]|\\.)*)'"
            match = re.search(failed_gen_pattern, error_str, re.DOTALL)
            
            if not match:
                # Method 2: Try with escaped quotes or different quote styles
                failed_gen_pattern = r'failed_generation[^:]*:\s*["\']((?:[^"\'\\]|\\.)*)["\']'
                match = re.search(failed_gen_pattern, error_str, re.DOTALL)
            
            # Method 3: Direct search - find function tag and query separately (most reliable)
            # First find the function name
            func_tag_match = re.search(r'<function=(\w+)', error_str)
            if func_tag_match:
                tool_name = func_tag_match.group(1)
                logger.debug(f"Found function tag: {tool_name}, now searching for query...")
                
                # Find the query value - it's in the format "query": "SELECT..."
                # We need to find the query value that comes after "query": "
                query_pattern = r'"query":\s*"((?:[^"\\]|\\.)*)"'
                query_match = re.search(query_pattern, error_str, re.DOTALL)
                if query_match:
                    query = query_match.group(1)
                    # Unescape
                    query = query.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                    query = query.rstrip(';').strip()
                    if query and query.upper().startswith('SELECT'):
                        logger.info(f"✅ Extracted SQL query from Groq error (method 3): {query[:100]}...")
                        logger.debug(f"Extracted SQL from Groq error: {tool_name}")
                        return {"tool_name": tool_name, "args": {"query": query}}
            
            # Method 4: Even more permissive - find function tag and extract query separately
            # This handles cases where the JSON structure might be malformed
            func_tag_match = re.search(r'<function=(\w+)', error_str)
            if func_tag_match:
                tool_name = func_tag_match.group(1)
                logger.debug(f"Found function tag: {tool_name}, now searching for query...")
                
                # Now find the query anywhere in the error string
                # Look for "query": "SELECT..." pattern
                query_match = re.search(r'"query":\s*"((?:[^"\\]|\\.)*)"', error_str, re.DOTALL)
                if query_match:
                    query = query_match.group(1)
                    query = query.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                    query = query.rstrip(';').strip()
                    if query and query.upper().startswith('SELECT'):
                        logger.info(f"✅ Extracted SQL query from Groq error (permissive method): {query[:100]}...")
                        logger.debug(f"Extracted SQL from Groq error: {tool_name}")
                        return {"tool_name": tool_name, "args": {"query": query}}
            
            if match:
                failed_gen_content = match.group(1)
                logger.debug(f"Extracted failed_generation content: {failed_gen_content[:200]}...")
                
                # Extract tool name and args from: <function=TOOL_NAME{...}></function>
                func_pattern = r'<function=(\w+)\s*({.*?})></function>'
                func_match = re.search(func_pattern, failed_gen_content, re.DOTALL)
                
                if func_match:
                    tool_name = func_match.group(1)
                    args_str = func_match.group(2)
                    logger.debug(f"Extracted tool_name: {tool_name}, args_str: {args_str[:200]}...")
                    
                    # Parse the JSON-like args string
                    try:
                        # The args might have escaped quotes, so we need to handle that carefully
                        # First, try direct JSON parsing
                        args = json.loads(args_str)
                        
                        logger.info(f"✅ Extracted tool call from Groq error: {tool_name} with args")
                        logger.debug(f"Extracted tool call from Groq error: {tool_name}")
                        return {"tool_name": tool_name, "args": args}
                    except json.JSONDecodeError as je:
                        logger.debug(f"JSON decode failed: {je}, trying regex extraction")
                        # If JSON parsing fails, try to extract query directly using regex
                        # The query is inside double quotes, so we need to handle escaped quotes
                        query_pattern = r'"query":\s*"((?:[^"\\]|\\.)*)"'
                        query_match = re.search(query_pattern, args_str, re.DOTALL)
                        
                        if query_match:
                            query = query_match.group(1)
                            # Unescape common escape sequences
                            query = query.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t')
                            # Remove trailing semicolon if present
                            query = query.rstrip(';').strip()
                            if query and query.upper().startswith('SELECT'):
                                logger.info(f"✅ Extracted SQL query from Groq error (regex): {query[:100]}...")
                                logger.debug(f"Extracted SQL from Groq error: {tool_name}")
                                return {"tool_name": tool_name, "args": {"query": query}}
                        else:
                            logger.warning(f"Could not extract query from args_str: {args_str[:200]}")
        except Exception as e:
            logger.warning(f"Failed to extract SQL from Groq error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        logger.warning("Failed to extract SQL from Groq error - no match found")
        return None
    
    def _create_synthetic_tool_response(self, tool_name: str, tool_args: Dict[str, Any], messages: List) -> AIMessage:
        """
        Create a synthetic AIMessage with tool_calls from extracted Groq error.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            messages: Current message history
            
        Returns:
            AIMessage with tool_calls
        """
        # Generate a unique tool call ID
        import uuid
        tool_call_id = str(uuid.uuid4())
        
        # Create tool call structure
        tool_call = {
            "id": tool_call_id,
            "name": tool_name,
            "args": tool_args
        }
        
        # Create AIMessage with tool calls
        synthetic_response = AIMessage(
            content="",
            tool_calls=[tool_call]
        )
        
        logger.info(f"✅ Created synthetic AIMessage with tool call: {tool_name}")
        logger.debug(f"Created synthetic response with tool call: {tool_name}")
        
        return synthetic_response
    
    def _extract_sql_from_text(self, text: str) -> Optional[str]:
        """
        Extract SQL query from markdown code blocks or plain text.
        
        Args:
            text: Text content that may contain SQL query
            
        Returns:
            Extracted SQL query string, or None if not found
        """
        if not text:
            return None
        
        logger.debug(f"Attempting to extract SQL from text (length: {len(text)} chars)")
        
        # Try markdown code blocks first (```sql ... ```)
        sql_pattern = r'```sql\s*(.*?)```'
        matches = re.findall(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            sql = matches[0].strip()
            # Remove any leading/trailing whitespace and newlines
            sql = re.sub(r'^\s+|\s+$', '', sql, flags=re.MULTILINE)
            if sql and sql.upper().startswith('SELECT'):
                logger.info(f"✅ Extracted SQL from markdown code block: {sql[:100]}...")
                return sql
        
        # Try markdown code blocks without language specifier (``` ... ```)
        sql_pattern = r'```\s*(SELECT.*?)```'
        matches = re.findall(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            sql = matches[0].strip()
            if sql.upper().startswith('SELECT'):
                logger.info(f"✅ Extracted SQL from generic code block: {sql[:100]}...")
                return sql
        
        # Try plain SQL (SELECT statements)
        sql_pattern = r'(SELECT\s+.*?(?:LIMIT\s+\d+)?(?:\s*;)?)'
        matches = re.findall(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            sql = matches[0].strip().rstrip(';')
            if sql:
                logger.info(f"✅ Extracted SQL from plain text: {sql[:100]}...")
                return sql
        
        logger.debug("No SQL query found in text")
        return None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Create tool node for executing tools
        tool_node = ToolNode(self.tools)
        
        # Wrap tool node to add logging and preserve message history
        def logged_tool_node(state: AgentState) -> AgentState:
            """Wrapper for tool node with logging and message preservation."""
            messages = state.get("messages", [])
            last_message = messages[-1] if messages else None
            user_question = state.get("question", "")
            
            # CRITICAL: Preserve the full message history before tool execution
            # The ToolNode will only return ToolMessages, so we need to keep everything
            preserved_messages = messages.copy()
            
            # Extract from_facility from question if present (for journey tools)
            from_facility = None
            if user_question:
                # Pattern: "starting from facility X" or "from facility X"
                import re
                patterns = [
                    r'starting\s+from\s+facility\s+([A-Z0-9]+)',
                    r'from\s+facility\s+([A-Z0-9]+)',
                    r'journeys?\s+from\s+facility\s+([A-Z0-9]+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, user_question, re.IGNORECASE)
                    if match:
                        from_facility = match.group(1).strip()
                        logger.debug(f"Extracted from_facility: {from_facility}")
                        break
            
            if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                for tc in last_message.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    if from_facility and tool_name in ["journey_list_tool", "journey_count_tool"]:
                        if not tool_args.get("params"):
                            tool_args["params"] = {}
                        if "from_facility" not in tool_args["params"]:
                            tool_args["params"]["from_facility"] = from_facility
                            tc["args"] = tool_args
                t_tools = time.perf_counter()
                result = tool_node.invoke(state)
                elapsed_tools = time.perf_counter() - t_tools
                logger.info(f"process=tool_exec time={elapsed_tools:.2f}s tools={tool_names}")
                _breakdown = state.get("stage_breakdown", [])
                _breakdown.append({"stage": "tool_exec", "elapsed_s": round(elapsed_tools, 3), "in": 0, "out": 0, "total": 0, "tools": tool_names})
                result["stage_breakdown"] = _breakdown
            
            new_tool_messages = result.get("messages", [])
            combined_messages = preserved_messages + new_tool_messages
            
            # Return state with complete message history
            return {
                **result,
                "messages": combined_messages
            }
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", logged_tool_node)
        workflow.add_node("format_answer", self._format_answer)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges - check if agent wants to call tools
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": "format_answer"
            }
        )
        
        # After tools, check if we have query results - if yes, go to format_answer directly
        # This prevents another LLM call with full history just to format the answer
        workflow.add_conditional_edges(
            "tools",
            self._should_format_after_tools,
            {
                "format": "format_answer",
                "continue": "agent"
            }
        )
        
        # Add edges
        workflow.add_edge("format_answer", END)
        
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Agent node that decides what to do next."""
        messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)
        
        # Check for column errors in ToolMessages and automatically call get_table_structure
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query"]:
                result_content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # Detect column errors (UndefinedColumn)
                if isinstance(result_content, str) and "UndefinedColumn" in result_content and "column" in result_content.lower() and "does not exist" in result_content.lower():
                    sql_query = state.get("sql_query", "")
                    if sql_query:
                        # Extract table name from SQL query
                        alias_match = re.search(r'FROM\s+(\w+)\s+(\w+)', sql_query, re.IGNORECASE)
                        if alias_match:
                            table_name = alias_match.group(1)
                            
                            # Check if we've already called get_table_structure for this table
                            already_called = any(
                                isinstance(m, ToolMessage) and m.name == "get_table_structure" and table_name in (m.content if hasattr(m, 'content') else str(m))
                                for m in messages
                            )
                            
                            # Only auto-call if we haven't already and haven't exceeded max iterations
                            if iteration_count < 4 and not already_called:
                                logger.info(f"🔍 Agent node: Detected column error for table {table_name}, auto-calling get_table_structure")
                                logger.debug(f"Auto-calling get_table_structure for {table_name}")
                                
                                try:
                                    table_structure_result = self.get_table_structure_tool.invoke({"table_names": table_name})
                                    
                                    # OpenAI (and strict APIs) require every ToolMessage to follow an AIMessage
                                    # that issued a matching tool_call. Add a synthetic AIMessage so the
                                    # auto-injected get_table_structure ToolMessage is valid.
                                    auto_tool_call_id = "auto_table_structure_002"
                                    synthetic_ai = AIMessage(
                                        content="",
                                        tool_calls=[{
                                            "id": auto_tool_call_id,
                                            "name": "get_table_structure",
                                            "args": {"table_names": table_name},
                                        }]
                                    )
                                    messages.append(synthetic_ai)
                                    structure_message = ToolMessage(
                                        content=table_structure_result,
                                        name="get_table_structure",
                                        tool_call_id=auto_tool_call_id
                                    )
                                    messages.append(structure_message)
                                    state["messages"] = messages
                                    
                                    logger.debug(f"Retrieved table structure for {table_name}")
                                except Exception as e:
                                    logger.error(f"Error calling get_table_structure in agent node: {e}")
                                    import traceback
                                    traceback.print_exc()
        
        # Ensure we have SystemMessage and HumanMessage
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        has_human = any(isinstance(m, HumanMessage) for m in messages)
        
        # Check if user query validation has been completed
        query_validated = state.get("query_validated", False)
        
        if not messages or not has_system or not has_human:
            # FIRST: Validate user query with security guard LLM call (saves tokens)
            # Only validate for non-admin users (admin can query across all users)
            if not query_validated and str(self.user_id).lower() != "admin":
                t_security = time.perf_counter()
                # OPTIMIZATION 4: Optimized security guard prompt - more concise
                security_guard_prompt = f"""Security Guard. User ID: {self.user_id}

SAFE (ALLOW): Device/facility queries, journeys, metrics for own data.
RISKY (BLOCK): Requests for other users' data, system schema, admin logs.

Device IDs (WT01...) and Facility IDs (MNIAZ...) are SAFE - not user IDs.
User IDs are standalone numbers (27, 63) in context of "user 27".

Respond ONLY: 'ALLOW' or 'BLOCK'"""
                
                user_question = state["question"]
                security_messages = [
                    SystemMessage(content=security_guard_prompt),
                    HumanMessage(content=user_question)
                ]
                
                security_response = self._invoke_llm_with_fallback(security_messages, use_tools=False, question=user_question)
                security_decision = security_response.content.strip().upper() if hasattr(security_response, 'content') else ""
                elapsed_security = time.perf_counter() - t_security
                security_token_usage = {"input": 0, "output": 0, "total": 0}
                if hasattr(security_response, "response_metadata") and security_response.response_metadata:
                    usage = security_response.response_metadata.get("token_usage", {})
                    if usage:
                        security_token_usage = {
                            "input": usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0),
                            "total": usage.get("total_tokens", 0)
                        }
                logger.info(f"process=security_guard time={elapsed_security:.2f}s in={security_token_usage['input']} out={security_token_usage['output']} total={security_token_usage['total']}")
                _breakdown = state.get("stage_breakdown", [])
                _breakdown.append({"stage": "security_guard", "elapsed_s": round(elapsed_security, 3), "in": security_token_usage["input"], "out": security_token_usage["output"], "total": security_token_usage["total"]})
                state["stage_breakdown"] = _breakdown

                # Check if query is blocked
                if "BLOCK" in security_decision:
                    error_msg = "Sorry, I cannot provide that information."
                    logger.warning("User query BLOCKED by Security Guard")
                    
                    # Update token usage
                    current_token_usage = state.get("token_usage", {"input": 0, "output": 0, "total": 0})
                    updated_token_usage = {
                        "input": current_token_usage.get("input", 0) + security_token_usage["input"],
                        "output": current_token_usage.get("output", 0) + security_token_usage["output"],
                        "total": current_token_usage.get("total", 0) + security_token_usage["total"]
                    }
                    
                    # Record security guard call in history
                    llm_call_history = state.get("llm_call_history", [])
                    llm_call_history.append({
                        "iteration": 1,
                        "call_type": "security_guard",
                        "input_messages": [{
                            "type": "SystemMessage",
                            "content_length": len(security_guard_prompt),
                            "content_preview": security_guard_prompt[:200] + "...",
                            "note": "Security guard prompt - minimal validation only"
                        }, {
                            "type": "HumanMessage",
                            "content": user_question
                        }],
                        "input_message_count": 2,
                        "output": {
                            "content": security_decision,
                            "has_tool_calls": False
                        },
                        "token_usage": security_token_usage
                    })
                    
                    return {
                        **state,
                        "final_answer": error_msg,
                        "query_validated": True,
                        "token_usage": updated_token_usage,
                        "llm_call_history": llm_call_history,
                        "messages": [HumanMessage(content=state["question"])],
                        "stage_breakdown": state.get("stage_breakdown", [])
                    }
                else:
                    # Mark as validated and continue with full prompt setup
                    state["query_validated"] = True
                    
                    # Record security guard call in history (even for allowed queries)
                    llm_call_history = state.get("llm_call_history", [])
                    llm_call_history.append({
                        "iteration": 1,
                        "call_type": "security_guard",
                        "input_messages": [{
                            "type": "SystemMessage",
                            "content_length": len(security_guard_prompt),
                            "content_preview": security_guard_prompt[:200] + "...",
                            "note": "Security guard prompt - minimal validation only"
                        }, {
                            "type": "HumanMessage",
                            "content": user_question
                        }],
                        "input_message_count": 2,
                        "output": {
                            "content": security_decision,
                            "has_tool_calls": False
                        },
                        "token_usage": security_token_usage
                    })
                    state["llm_call_history"] = llm_call_history
                
                # Update token usage from security guard call
                current_token_usage = state.get("token_usage", {"input": 0, "output": 0, "total": 0})
                state["token_usage"] = {
                    "input": current_token_usage.get("input", 0) + security_token_usage["input"],
                    "output": current_token_usage.get("output", 0) + security_token_usage["output"],
                    "total": current_token_usage.get("total", 0) + security_token_usage["total"]
                }
            elif not query_validated and str(self.user_id).lower() == "admin":
                state["query_validated"] = True
                logger.info("process=security_guard time=0s skipped=admin")
                _breakdown = state.get("stage_breakdown", [])
                _breakdown.append({"stage": "security_guard", "elapsed_s": 0.0, "in": 0, "out": 0, "total": 0, "note": "skipped (admin)"})
                state["stage_breakdown"] = _breakdown
            
            # Determine if this is a journey question for prompt optimization
            is_journey = self._is_journey_question(state["question"])
            
            # STEP 2: Initialize messages with FULL prompt (only if query is validated)
            t_vector = time.perf_counter()
            system_prompt = get_system_prompt(
                user_id=self.user_id,
                top_k=self.top_k,
                question=state["question"],
                vector_store_manager=self.vector_store_manager,
                preload_examples=True,
                is_journey=is_journey
            )
            elapsed_vector = time.perf_counter() - t_vector
            logger.info(f"process=vector_search time={elapsed_vector:.2f}s")
            _breakdown = state.get("stage_breakdown", [])
            _breakdown.append({"stage": "vector_search", "elapsed_s": round(elapsed_vector, 3), "in": 0, "out": 0, "total": 0})
            state["stage_breakdown"] = _breakdown
            base_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["question"])
            ]
            
            # Add any existing AIMessages and ToolMessages
            existing_ai = [m for m in messages if isinstance(m, AIMessage)]
            existing_tools = [m for m in messages if isinstance(m, ToolMessage)]
            
            messages = base_messages + existing_ai + existing_tools
            _log_messages_debug(messages, "Initial Messages Created (After Validation)")
        else:
            # Messages are already in good shape, just validate the sequence
            # Validate message order: ToolMessages must follow their corresponding AIMessage
            valid_messages = []
            for i, msg in enumerate(messages):
                if isinstance(msg, (SystemMessage, HumanMessage, AIMessage)):
                    valid_messages.append(msg)
                elif isinstance(msg, ToolMessage):
                    # Check if there's a preceding AIMessage with matching tool_call_id
                    tool_call_id = getattr(msg, "tool_call_id", None)
                    has_matching_ai = False
                    
                    # Look backwards for matching AIMessage
                    for j in range(len(valid_messages) - 1, -1, -1):
                        prev_msg = valid_messages[j]
                        if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, "tool_calls"):
                            tool_call_ids = [tc.get("id") for tc in prev_msg.tool_calls if tc.get("id")]
                            if tool_call_id and tool_call_id in tool_call_ids:
                                valid_messages.append(msg)
                                has_matching_ai = True
                                break
                    
                    if not has_matching_ai:
                        valid_messages.append(msg)
            
            messages = valid_messages
        
        # Filter out check_user_query_restriction tool messages from history
        # This tool is only for validation and shouldn't be included in subsequent LLM calls
        # to reduce token consumption
        filtered_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Check if this is a check_user_query_restriction tool message
                tool_name = getattr(msg, 'name', '')
                if tool_name == 'check_user_query_restriction':
                    continue
            filtered_messages.append(msg)
        
        messages = filtered_messages
        
        # Invoke LLM with tools
        try:
            if not messages:
                raise ValueError("Cannot invoke LLM with empty messages array")
            
            _log_messages_debug(messages, f"Before LLM Invocation (Iteration {state.get('iteration_count', 0) + 1})")
            question = state.get("question", "")
            self._bind_tools_conditionally(question)
            model_name = getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model', None) or 'Unknown'
            t_agent = time.perf_counter()
            
            # Track what messages are being sent TO the LLM
            iteration = state.get("iteration_count", 0) + 1
            llm_call_history = state.get("llm_call_history", [])
            
            # Build input messages summary for this call
            input_messages_summary = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    input_messages_summary.append({
                        "type": "SystemMessage",
                        "content_length": len(msg.content),
                        "content_preview": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    })
                elif isinstance(msg, HumanMessage):
                    input_messages_summary.append({
                        "type": "HumanMessage",
                        "content": msg.content
                    })
                elif isinstance(msg, AIMessage):
                    input_messages_summary.append({
                        "type": "AIMessage",
                        "content": msg.content if msg.content else None,
                        "has_tool_calls": bool(hasattr(msg, "tool_calls") and msg.tool_calls)
                    })
                elif isinstance(msg, ToolMessage):
                    input_messages_summary.append({
                        "type": "ToolMessage",
                        "name": getattr(msg, "name", "unknown"),
                        "content_length": len(msg.content),
                        "content_preview": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    })
            
            try:
                if self.llm_with_tools is None:
                    self._bind_tools_conditionally(question if question else "")
                response = self._invoke_llm_with_fallback(messages, use_tools=True, question=question)
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                logger.error(f"LLM failed: {error_type} - {error_str[:200]}")
                logger.exception("Full exception")
                raise e
            
            elapsed_agent = time.perf_counter() - t_agent
            messages.append(response)
            
            token_usage = state.get("token_usage", {"input": 0, "output": 0, "total": 0})
            call_token_usage = {"input": 0, "output": 0, "total": 0}
            if hasattr(response, "response_metadata") and response.response_metadata:
                usage = response.response_metadata.get("token_usage", {})
                if usage:
                    call_token_usage = {
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0)
                    }
                    token_usage["input"] = token_usage.get("input", 0) + call_token_usage["input"]
                    token_usage["output"] = token_usage.get("output", 0) + call_token_usage["output"]
                    token_usage["total"] = token_usage.get("total", 0) + call_token_usage["total"]
            logger.info(f"process=agent_node time={elapsed_agent:.2f}s in={call_token_usage['input']} out={call_token_usage['output']} total={call_token_usage['total']}")
            _breakdown = state.get("stage_breakdown", [])
            _breakdown.append({"stage": "agent_node", "elapsed_s": round(elapsed_agent, 3), "in": call_token_usage["input"], "out": call_token_usage["output"], "total": call_token_usage["total"]})
            state["stage_breakdown"] = _breakdown

            # Record this LLM call in history
            llm_call_info = {
                "iteration": iteration,
                "call_type": "agent_node",
                "input_messages": input_messages_summary,
                "input_message_count": len(messages),
                "output": {
                    "content": response.content if response.content else None,
                    "has_tool_calls": bool(hasattr(response, "tool_calls") and response.tool_calls),
                    "tool_calls": []
                },
                "token_usage": call_token_usage
            }
            
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls_info = []
                for tc in response.tool_calls:
                    tool_call_info = {
                        "name": tc.get("name"),
                        "args": tc.get("args", {})
                    }
                    # Extract SQL query if it's a query tool or journey tool
                    tool_name = tc.get("name")
                    if tool_name in ["execute_db_query", "count_query", "list_query"]:
                        sql_query = tc.get("args", {}).get("query", "")
                        tool_call_info["sql_query"] = sql_query
                    elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                        # Journey tools use "sql" parameter
                        sql_query = tc.get("args", {}).get("sql", "")
                        tool_call_info["sql_query"] = sql_query
                    tool_calls_info.append(tool_call_info)
                llm_call_info["output"]["tool_calls"] = tool_calls_info
            
            llm_call_history.append(llm_call_info)
            
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [tc.get('name') for tc in response.tool_calls]
                logger.info(f"process=agent_node tool_calls={tool_names}")
            else:
                # Try to extract SQL from text response when tool calls fail
                extracted_sql = None
                if response.content:
                    extracted_sql = self._extract_sql_from_text(response.content)
                if extracted_sql:
                    logger.info("process=agent_node extracted_sql_from_text")
                    try:
                        query_result = self.query_tool_instance.execute(extracted_sql)
                        tool_message = ToolMessage(
                            content=query_result,
                            name="execute_db_query",
                            tool_call_id="manual_extraction_001"
                        )
                        messages.append(tool_message)
                        state["sql_query"] = extracted_sql
                        state["query_result"] = query_result
                    except Exception as exec_error:
                        logger.error(f"Error executing extracted SQL: {exec_error}")
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            logger.exception("LLM invocation")
            raise
        
        # Extract SQL query if execute_db_query or journey tools were called
        sql_query = state.get("sql_query")
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.debug(f"Extracting SQL from {len(response.tool_calls)} tool calls")
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                if tool_name in ["execute_db_query", "count_query", "list_query"]:
                    sql_query = tool_call["args"].get("query", "")
                    break
                elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                    sql_query = tool_call["args"].get("sql", "")
                    break
        
        # Check if we got query results from previous tool execution
        query_result = state.get("query_result")
        logger.debug(f"Current state - SQL query: {bool(sql_query)}, Query result: {bool(query_result)}")
        if not query_result:
            # Look for tool messages with query results (check all query tools)
            for msg in messages:
                if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query"]:
                    query_result = msg.content
                    break
        
        # Increment iteration count
        iteration_count = state.get("iteration_count", 0) + 1
        
        # Update token usage in state
        current_token_usage = state.get("token_usage", {"input": 0, "output": 0, "total": 0})
        if hasattr(response, "response_metadata") and response.response_metadata:
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                current_token_usage = {
                    "input": current_token_usage.get("input", 0) + usage.get("prompt_tokens", 0),
                    "output": current_token_usage.get("output", 0) + usage.get("completion_tokens", 0),
                    "total": current_token_usage.get("total", 0) + usage.get("total_tokens", 0)
                }
        
        return {
            **state,
            "messages": messages,
            "sql_query": sql_query or state.get("sql_query"),
            "query_result": query_result or state.get("query_result"),
            "iteration_count": iteration_count,
            "token_usage": current_token_usage,
            "llm_call_history": llm_call_history
        }
    
    def _should_format_after_tools(self, state: AgentState) -> str:
        """
        Check if we should format answer directly after tool execution.
        If we have query results, skip the agent node and go directly to format_answer.
        This saves tokens by avoiding another LLM call with full message history.
        
        However, if the result is a column error, automatically call get_table_structure
        and route back to agent to retry with correct schema.
        """
        messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)
        
        # Check if we have query results from any query tool (including journey tools)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                # Check if the result is an error about undefined columns
                result_content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # Detect column errors (UndefinedColumn) - route back to agent to handle
                if isinstance(result_content, str) and "UndefinedColumn" in result_content and "column" in result_content.lower() and "does not exist" in result_content.lower():
                    # Extract table name from SQL query to check if we should retry
                    sql_query = state.get("sql_query", "")
                    if sql_query:
                        alias_match = re.search(r'FROM\s+(\w+)\s+(\w+)', sql_query, re.IGNORECASE)
                        if alias_match:
                            table_name = alias_match.group(1)
                            
                            # Check if we've already called get_table_structure for this table
                            already_called = any(
                                isinstance(m, ToolMessage) and m.name == "get_table_structure" and table_name in (m.content if hasattr(m, 'content') else str(m))
                                for m in messages
                            )
                            
                            # Only retry if we haven't exceeded max iterations and haven't already called get_table_structure
                            if iteration_count < 4 and not already_called:
                                logger.info(f"[column_error] table={table_name} → retry agent")
                                return "continue"
                            else:
                                if already_called:
                                    logger.warning(f"Already called get_table_structure for {table_name}, formatting error message instead")
                                else:
                                    logger.warning(f"Max iterations reached ({iteration_count}), formatting error message instead")
                                # Fall through to format the error
                
                logger.info(f"process=tool_result tool={msg.name} next=format_answer")
                return "format"
        
        # No query results yet, continue to agent
        logger.debug("No tool results found, continuing to agent")
        return "continue"
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue (call tools) or end."""
        # Check if final_answer is already set (e.g., from early validation rejection)
        final_answer = state.get("final_answer")
        if final_answer:
            return "end"
        
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        iteration_count = state.get("iteration_count", 0)
        
        if iteration_count >= 5:
            logger.warning(f"Max iterations reached ({iteration_count}), forcing end")
            return "end"
        
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.debug(f"Decision: CONTINUE tool_calls={len(last_message.tool_calls)}")
            return "continue"
        
        query_result = state.get("query_result")
        if query_result and query_result != "":
            return "end"
        
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                return "end"
        
        if state.get("sql_query"):
            for msg in messages:
                if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                    return "end"
        
        if last_message and isinstance(last_message, AIMessage):
            if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
                return "end"
        
        return "continue"
    
    def _format_answer(self, state: AgentState) -> AgentState:
        """Format final natural language answer."""
        t_format = time.perf_counter()
        messages = state.get("messages", [])
        _log_messages_debug(messages, "All Messages Before Formatting")
        
        # Extract query result from messages if not in state
        query_result = state.get("query_result", "")
        journey_tool_used = False
        
        # CRITICAL: Check if query_result in state is an error - if so, clear it and re-extract
        if query_result and isinstance(query_result, str):
            is_error_in_state = (
                query_result.startswith("Error executing query") or
                "UndefinedColumn" in query_result or
                "does not exist" in query_result.lower() or
                query_result.startswith("::::::")
            )
            if is_error_in_state:
                logger.warning(f"⚠️  State contains error message, clearing and re-extracting from ToolMessages")
                query_result = ""  # Clear the error so we can find the successful result
        
        if not query_result:
            # Try to extract from tool messages (check all query tools, including journey tools)
            # IMPORTANT: Skip error messages and only use successful results
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', '')
                    if tool_name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        # Skip error messages - look for successful results only
                        if isinstance(content, str):
                            # Check if this is an error message
                            is_error = (
                                content.startswith("Error executing query") or
                                "UndefinedColumn" in content or
                                "does not exist" in content.lower() or
                                content.startswith("::::::")
                            )
                            
                            if not is_error:
                                query_result = content
                                if tool_name in ["journey_list_tool", "journey_count_tool"]:
                                    journey_tool_used = True
                                break
                            else:
                                # This is an error - skip it and continue looking
                                logger.debug(f"⏭️  Skipping error ToolMessage ({tool_name}): {content[:100]}...")
                                continue
                        else:
                            query_result = content
                            if tool_name in ["journey_list_tool", "journey_count_tool"]:
                                journey_tool_used = True
                            break
        
        # Extract SQL query from messages if not in state
        # IMPORTANT: Match SQL query to the successful result (not the error)
        sql_query = state.get("sql_query", "")
        if not sql_query:
            # Try to find SQL query in tool calls (check all query tools, including journey tools)
            # Match it to the successful ToolMessage if we found one
            logger.debug("SQL query not in state, searching in messages...")
            
            # First, find the successful ToolMessage to get its tool_call_id
            successful_tool_call_id = None
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', '')
                    if tool_name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        if isinstance(content, str):
                            is_error = (
                                content.startswith("Error executing query") or
                                "UndefinedColumn" in content or
                                "does not exist" in content.lower() or
                                content.startswith("::::::")
                            )
                            if not is_error:
                                successful_tool_call_id = getattr(msg, 'tool_call_id', None)
                                break
            
            # Now find the AIMessage with the matching tool_call_id
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        tool_call_id = tool_call.get("id", "")
                        # If we found a successful result, only use SQL from that tool call
                        if successful_tool_call_id and tool_call_id != successful_tool_call_id:
                            continue
                            
                        tool_name = tool_call.get("name", "")
                        if tool_name in ["execute_db_query", "count_query", "list_query"]:
                            sql_query = tool_call["args"].get("query", "")
                            break
                        elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                            sql_query = tool_call["args"].get("sql", "")
                            break
                    if sql_query:
                        break
            
            # Fallback: if no successful result found, just get the latest SQL query
            if not sql_query:
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name", "")
                            if tool_name in ["execute_db_query", "count_query", "list_query"]:
                                sql_query = tool_call["args"].get("query", "")
                                break
                            elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                                sql_query = tool_call["args"].get("sql", "")
                                break
                        if sql_query:
                            break
            
            # If still not found, try extracting from text messages
            if not sql_query:
                logger.debug("SQL query not found in tool calls, trying to extract from text...")
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        extracted = self._extract_sql_from_text(msg.content)
                        if extracted:
                            sql_query = extracted
                            break
        
        logger.debug(f"format_answer sql_query={bool(sql_query)} query_result_len={len(query_result or '')}")
        
        # If we have query results, format them
        # CRITICAL: Double-check that query_result is not an error message
        if query_result:
            # Final safety check: reject if it's still an error
            if isinstance(query_result, str):
                is_still_error = (
                    query_result.startswith("Error executing query") or
                    "UndefinedColumn" in query_result or
                    query_result.startswith("::::::")
                )
                if is_still_error:
                    logger.error(f"❌ CRITICAL: query_result is still an error after extraction! This should not happen.")
                    logger.error(f"   Error content: {query_result[:300]}...")
                    # Try one more time to find successful result
                    for msg in reversed(messages):
                        if isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, 'name', '')
                            if tool_name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                                content = msg.content if hasattr(msg, 'content') else str(msg)
                                if isinstance(content, str) and not (
                                    content.startswith("Error executing query") or
                                    "UndefinedColumn" in content or
                                    content.startswith("::::::")
                                ):
                                    query_result = content
                                    logger.debug(f"Recovered result from ToolMessage ({tool_name})")
                                    break
                    # If still an error, we can't format it
                    if isinstance(query_result, str) and (
                        query_result.startswith("Error executing query") or
                        "UndefinedColumn" in query_result
                    ):
                        logger.error(f"❌ Cannot format answer - only error messages found")
                        query_result = None
        
        if query_result and not query_result.startswith("::::::"):
            logger.info("process=format_answer start")
            
            # SAFETY CHECK: Truncate extremely large results before processing
            # LLM APIs have request size limits (typically 1-2M tokens, ~4-8M characters)
            # We'll limit to 50,000 characters to be safe
            MAX_RESULT_SIZE = 50000
            original_result_size = len(query_result) if query_result else 0
            
            if original_result_size > MAX_RESULT_SIZE:
                logger.warning(f"Query result too large ({original_result_size:,} chars), truncating for LLM")
                
                # For extremely large results, create a summary instead of trying to parse everything
                # This prevents memory issues and API request size errors
                try:
                    # Check if it's a list/dict format by looking at first character
                    if isinstance(query_result, str) and query_result.strip():
                        first_char = query_result.strip()[0]
                        
                        if first_char == '[':
                            # It's a list - count approximate rows by counting '},' or '],' patterns
                            # This is much faster than parsing 92M characters
                            row_indicators = query_result.count('},') + query_result.count('],')
                            estimated_count = row_indicators + 1  # +1 for last item
                            
                            # Try to get first few items (parse only first 10KB)
                            try:
                                import ast
                                sample_str = query_result[:10000]  # First 10KB should have a few items
                                # Find the end of first complete item
                                bracket_count = 0
                                sample_end = 0
                                for i, char in enumerate(sample_str):
                                    if char == '[':
                                        bracket_count += 1
                                    elif char == ']':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            sample_end = i + 1
                                            break
                                
                                if sample_end > 0:
                                    parsed_sample = ast.literal_eval(sample_str[:sample_end] + ']')
                                    if isinstance(parsed_sample, list) and len(parsed_sample) > 0:
                                        sample = parsed_sample[:3]  # First 3 items
                                        query_result = f"Total results: ~{estimated_count:,} rows (estimated)\n\nFirst 3 rows (sample):\n{json.dumps(sample, indent=2, default=str)}\n\n[Result too large to display fully ({original_result_size:,} chars). Please refine your query with more specific filters or time ranges.]"
                                    else:
                                        query_result = f"Total results: ~{estimated_count:,} rows (estimated)\n\n[Result too large to display ({original_result_size:,} chars). Please refine your query with more specific filters or time ranges.]"
                                else:
                                    query_result = f"Total results: ~{estimated_count:,} rows (estimated)\n\n[Result too large to display ({original_result_size:,} chars). Please refine your query with more specific filters or time ranges.]"
                            except Exception:
                                # If parsing fails, just show count estimate
                                query_result = f"Total results: ~{estimated_count:,} rows (estimated)\n\n[Result too large to display ({original_result_size:,} chars). Please refine your query with more specific filters or time ranges.]"
                        
                        elif first_char == '{':
                            # It's a dict - just truncate with summary
                            query_result = query_result[:MAX_RESULT_SIZE] + f"\n\n[Result truncated - too large to display ({original_result_size:,} chars). Please refine your query with more specific filters or time ranges.]"
                        else:
                            # String format - just truncate
                            query_result = query_result[:MAX_RESULT_SIZE] + f"\n\n[Result truncated - too large to display ({original_result_size:,} chars). Please refine your query with more specific filters or time ranges.]"
                    else:
                        # Not a string or empty - convert and truncate
                        query_result = str(query_result)[:MAX_RESULT_SIZE] + f"\n\n[Result truncated - too large to display ({original_result_size:,} chars). Please refine your query with more specific filters or time ranges.]"
                        
                except Exception as e:
                    logger.warning(f"Error creating summary for large result: {e}. Using simple truncation.")
                    # Final fallback: simple truncation
                    query_result = str(query_result)[:MAX_RESULT_SIZE] + f"\n\n[Result truncated - too large to display ({original_result_size:,} chars). Please refine your query with more specific filters or time ranges.]"
            
            # Generate final answer from results using a MINIMAL, one-shot prompt.
            # IMPORTANT: We DO NOT send the full message history, system prompt,
            # or retrieved examples again here. This keeps token usage low for
            # the final "explain the result" step.
            user_question = state.get("question", "")
            user_id = state.get("user_id", "")
            
            # For journey tools, add special instruction
            journey_instruction = ""
            if journey_tool_used:
                journey_instruction = """
                
                IMPORTANT: The results are from journey calculation.
                
                For journey LISTS (journies array):
                - If _showing_preview is true, only a sample is shown. Mention the total count and CSV link.
                - If user asked for journeys "starting from facility X", the results are already filtered.
                
                For journey COUNTS (journey_details object):
                - If user asks about facility type transitions (e.g., "M to R"):
                  * Manufacturer types: "D" and "M"
                  * Retailer type: "R"
                  * Filter journey_details where from_type is "D" or "M" AND to_type is "R"
                  * Sum the "count" values from matching entries
                
                Be concise. Focus on answering the specific question asked.
                """

            # Extract CSV download link from query_result if present
            csv_download_url = None
            csv_path = None
            csv_id = None
            
            # Try to parse as JSON first (for journey tools)
            result_dict = None
            try:
                if query_result.strip().startswith('{'):
                    result_dict = json.loads(query_result)
                    # Check for CSV link in journey results
                    if 'csv_download_link' in result_dict:
                        csv_path = result_dict['csv_download_link']
                        csv_id = result_dict.get('csv_id')
                        csv_download_url = f"{settings.get_api_base_url()}{csv_path}"
                        logger.debug(f"Extracted CSV link from JSON")
            except (json.JSONDecodeError, KeyError, TypeError):
                # Not JSON or no CSV link, try text format
                pass
            
            # Also check for CSV link in text format (for regular SQL queries with CSV)
            if not csv_download_url:
                # Look for "CSV Download Link: /download-csv/..." pattern
                csv_link_match = re.search(r'CSV Download Link:\s*(/download-csv/[^\s\n]+)', query_result, re.IGNORECASE)
                if csv_link_match:
                    csv_path = csv_link_match.group(1)
                    csv_download_url = f"{settings.get_api_base_url()}{csv_path}"
                    logger.debug(f"Extracted CSV link from text")
                    
                    # Also try to extract CSV ID if present
                    csv_id_match = re.search(r'CSV ID:\s*([a-f0-9\-]+)', query_result, re.IGNORECASE)
                    if csv_id_match:
                        csv_id = csv_id_match.group(1)
                        logger.debug(f"Extracted CSV ID: {csv_id}")
            
            # For journey tools, reduce token usage by filtering/truncating results before format_answer
            if journey_tool_used and result_dict:
                journies = result_dict.get('journies', [])
                # IMPORTANT: Use total_journeys from result_dict if available (when CSV is generated),
                # otherwise use len(journies). This ensures we show the correct total count (58) not preview count (5)
                total_journeys = result_dict.get('total_journeys', len(journies))
                
                # Check if CSV is available (either from extracted csv_download_url or in result_dict)
                has_csv = csv_download_url or result_dict.get('csv_download_link') or result_dict.get('csv_id')
                
                # If CSV is available OR >5 journeys, use minimal summary only
                if has_csv or total_journeys > 5:
                    # Get CSV link from result_dict if not already extracted
                    if not csv_download_url:
                        csv_path = result_dict.get('csv_download_link')
                        if csv_path:
                            csv_download_url = f"{settings.get_api_base_url()}{csv_path}"
                            csv_id = result_dict.get('csv_id')
                    
                    # Create minimal summary - NO full data, just counts and CSV link
                    # This dramatically reduces token usage (from 44K chars to ~200 chars)
                    minimal_summary = {
                        "total_journeys": total_journeys,  # This is the ACTUAL total (58), not preview count (5)
                        "csv_download_link": result_dict.get('csv_download_link') or (csv_download_url.replace(settings.get_api_base_url(), "") if csv_download_url else None),
                        "csv_id": result_dict.get('csv_id') or csv_id,
                        "note": f"Full data available in CSV. Showing summary only to reduce token usage."
                    }
                    
                    # Never include journey data when CSV is available (saves massive tokens)
                    # Only include example if no CSV and small result
                    if not has_csv and total_journeys <= 10 and len(journies) > 0:
                        minimal_summary["example_journey"] = journies[0]
                    
                    query_result = json.dumps(minimal_summary, indent=2, default=str)
                    logger.debug(f"Token optimization: minimal summary total_journeys={total_journeys} csv={has_csv}")
                elif total_journeys > 0:
                    # Small result set (<=5 journeys), but still reduce facilities_details
                    # Keep only facilities that appear in the journeys
                    facilities_in_journeys = set()
                    for journey in journies:
                        facilities_in_journeys.add(journey.get('from_facility'))
                        facilities_in_journeys.add(journey.get('to_facility'))
                    
                    # Filter facilities_details to only those in journeys
                    filtered_facilities = {
                        fid: details 
                        for fid, details in result_dict.get('facilities_details', {}).items()
                        if fid in facilities_in_journeys
                    }
                    result_dict['facilities_details'] = filtered_facilities
                    query_result = json.dumps(result_dict, indent=2, default=str)
                    logger.info(f"📊 Filtered facilities_details to {len(filtered_facilities)} facilities (from journeys only)")
            
            # If not found in JSON, try text format (for regular SQL queries)
            if not csv_download_url:
                csv_link_match = re.search(r'CSV Download Link:\s*(/download-csv/[^\s\n]+)', query_result)
                if csv_link_match:
                    csv_path = csv_link_match.group(1)
                    csv_download_url = f"{settings.get_api_base_url()}{csv_path}"
                    logger.debug(f"Extracted CSV link from text")
                    
                    # For regular SQL queries with CSV, also create minimal summary
                    # Extract row count from query_result if available
                    row_count_match = re.search(r'Total rows:\s*(\d+)', query_result)
                    if row_count_match:
                        row_count = int(row_count_match.group(1))
                        # Create minimal summary for regular SQL queries too
                        minimal_summary = f"""Total rows: {row_count}
CSV Download Link: {csv_path}
Note: Full data available in CSV. Showing summary only."""
                        query_result = minimal_summary
                        logger.debug(f"Token optimization: minimal summary rows={row_count}")

            # Build the prompt with CSV link if available (use relative path only for UI and Postman)
            csv_link_instruction = ""
            if csv_download_url and csv_path:
                csv_link_instruction = f"\n\nIMPORTANT: Add exactly ONE download link at the end. Do NOT write the words 'Download CSV' or 'using the following link' before the link. Mention the count, then add only this link: [Download CSV]({csv_path})"

            # OPTIMIZATION: Ultra-minimal prompt when CSV is available
            # Safety check: If CSV is available but query_result is still large, force minimal summary
            if csv_download_url and len(query_result) > 1000:
                # Force minimal summary if somehow we still have large data
                if journey_tool_used:
                    # For journey tools, extract just the count
                    try:
                        if isinstance(query_result, str) and query_result.strip().startswith('{'):
                            temp_dict = json.loads(query_result)
                            total = temp_dict.get('total_journeys', 'unknown')
                            csv_link = temp_dict.get('csv_download_link') or (csv_download_url.replace(settings.get_api_base_url(), "") if csv_download_url else "")
                            query_result = f'{{"total_journeys": {total}, "csv_download_link": "{csv_link}", "note": "Full data in CSV"}}'
                        else:
                            query_result = f'Total journeys available in CSV. Download link: {csv_download_url}'
                    except Exception as e:
                        logger.warning(f"Error parsing query_result in safety check: {e}")
                        query_result = f'Total journeys available in CSV. Download link: {csv_download_url}'
                else:
                    # For regular SQL, extract row count
                    row_match = re.search(r'Total rows:\s*(\d+)', query_result)
                    if row_match:
                        query_result = f'Total rows: {row_match.group(1)}. Full data in CSV: {csv_download_url}'
                    else:
                        query_result = f'Results available in CSV. Download link: {csv_download_url}'
                
                logger.warning(f"Forced minimal summary: query_result too large with CSV")
            
            if csv_download_url:
                # When CSV is available, use even more minimal prompt
                final_prompt = f"""User asked: {user_question}

Query results summary: {query_result}

IMPORTANT: Use the EXACT "total_journeys" count from the results (not the preview count). Mention this total number, then add only the download link (do not write 'using the following link' or repeat 'Download CSV' before the link).{csv_link_instruction}""".strip()
            else:
                # Regular prompt for small results
                final_prompt = f"""User asked: {user_question}

Query results: {query_result}

Provide a concise, natural language answer. Do not mention table names, SQL syntax, or schema details.{journey_instruction}""".strip()
            
            llm_call_history = state.get("llm_call_history", [])
            from langchain_core.messages import HumanMessage
            final_messages = [HumanMessage(content=final_prompt)]
            response = self._invoke_llm_with_fallback(final_messages, use_tools=False, question=user_question)
            final_answer = response.content if hasattr(response, 'content') else str(response)
            # Normalize CSV link: use relative path and remove duplicate "Download CSV" phrasing
            if csv_path:
                if settings.get_api_base_url():
                    base = settings.get_api_base_url().rstrip("/")
                    final_answer = re.sub(rf"\(({re.escape(base)}/download-csv/[a-f0-9\-]+)\)", f"({csv_path})", final_answer)
                # Remove redundant "using the following link: [Download CSV]" so only the link shows once
                final_answer = re.sub(
                    r"\s*[Uu]sing the following link:\s*\[Download CSV\]\s*",
                    " ",
                    final_answer,
                    flags=re.IGNORECASE,
                )
            elapsed_format = time.perf_counter() - t_format
            call_token_usage = {"input": 0, "output": 0, "total": 0}
            if hasattr(response, "response_metadata") and response.response_metadata:
                usage = response.response_metadata.get("token_usage", {})
                if usage:
                    call_token_usage = {
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0)
                    }
            logger.info(f"process=format_answer time={elapsed_format:.2f}s in={call_token_usage['input']} out={call_token_usage['output']} total={call_token_usage['total']}")
            _breakdown = state.get("stage_breakdown", [])
            _breakdown.append({"stage": "format_answer", "elapsed_s": round(elapsed_format, 3), "in": call_token_usage["input"], "out": call_token_usage["output"], "total": call_token_usage["total"]})
            state["stage_breakdown"] = _breakdown

            # Update state's token_usage with this call's tokens
            current_token_usage = state.get("token_usage", {"input": 0, "output": 0, "total": 0})
            updated_token_usage = {
                "input": current_token_usage.get("input", 0) + call_token_usage["input"],
                "output": current_token_usage.get("output", 0) + call_token_usage["output"],
                "total": current_token_usage.get("total", 0) + call_token_usage["total"]
            }
            
            # Record final answer LLM call in history
            llm_call_info = {
                "iteration": state.get("iteration_count", 0) + 1,
                "call_type": "format_answer",
                "input_messages": [{
                    "type": "HumanMessage",
                    "content": final_prompt,
                    "content_length": len(final_prompt),
                    "note": "Minimal prompt - only question, SQL, and results (no system prompt, no examples, no history)"
                }],
                "input_message_count": 1,
                "output": {
                    "content": final_answer,
                    "has_tool_calls": False
                },
                "token_usage": call_token_usage
            }
            llm_call_history.append(llm_call_info)
            out = {
                **state,
                "final_answer": final_answer,
                "query_result": query_result,  # Update state with successful result (not error)
                "sql_query": sql_query,  # Update state with correct SQL query
                "llm_call_history": llm_call_history,
                "token_usage": updated_token_usage
            }
            if csv_path:
                out["csv_download_path"] = csv_path
                out["csv_id"] = csv_id if csv_id else csv_path.replace("/download-csv/", "").strip()
            return out
        elif query_result and query_result.startswith("::::::"):
            # Empty result
            final_answer = f"Based on the query, there are no results matching your criteria for the question: {state['question']}"
        else:
            # No query executed - use last message
            logger.warning("No query result found, attempting to use last AI message")
            if messages:
                last_ai_message = None
                for msg in reversed(messages):
                    # Check if it's an AIMessage without tool_calls (or with empty tool_calls)
                    if isinstance(msg, AIMessage):
                        has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                        if not has_tool_calls:
                            last_ai_message = msg
                            break
                
                if last_ai_message:
                    logger.info("Using last AI message content as final answer")
                    logger.debug(f"Last AI message content length: {len(last_ai_message.content) if last_ai_message.content else 0}")
                    
                    # Also try to extract SQL from the message if not already extracted
                    if not sql_query and last_ai_message.content:
                        extracted_sql = self._extract_sql_from_text(last_ai_message.content)
                        if extracted_sql:
                            sql_query = extracted_sql
                            logger.info(f"✅ Extracted SQL from final message: {sql_query[:100]}...")
                            
                            # Try to execute it
                            try:
                                logger.info("Attempting to execute SQL from final message")
                                query_result = self.query_tool_instance.execute(extracted_sql)
                                state["sql_query"] = extracted_sql
                                state["query_result"] = query_result
                                logger.info(f"✅ Executed SQL from final message, result length: {len(query_result)}")
                                
                                # If we got results, format them
                                if query_result and not query_result.startswith("::::::"):
                                    # Generate final answer from results
                                    user_question = state.get("question", "")
                                    user_id = state.get("user_id", "")
                                    
                                    # Extract CSV download link from query_result if present
                                    csv_download_url = None
                                    csv_id = None
                                    
                                    # Try to parse as JSON first (for journey tools)
                                    try:
                                        if query_result.strip().startswith('{'):
                                            result_dict = json.loads(query_result)
                                            if 'csv_download_link' in result_dict:
                                                csv_path = result_dict['csv_download_link']
                                                csv_id = result_dict.get('csv_id')
                                                csv_download_url = f"{settings.get_api_base_url()}{csv_path}"
                                                logger.info(f"✅ Extracted CSV download link from JSON: {csv_download_url}")
                                    except (json.JSONDecodeError, KeyError, TypeError):
                                        pass
                                    
                                    # If not found in JSON, try text format
                                    if not csv_download_url:
                                        csv_link_match = re.search(r'CSV Download Link:\s*(/download-csv/[^\s\n]+)', query_result)
                                        if csv_link_match:
                                            csv_path = csv_link_match.group(1)
                                            csv_download_url = f"{settings.get_api_base_url()}{csv_path}"
                                            logger.info(f"✅ Extracted CSV download link from text: {csv_download_url}")

                                    # Build the prompt with CSV link if available (relative path only for UI/Postman)
                                    csv_link_instruction = ""
                                    if csv_download_url and csv_path:
                                        csv_link_instruction = f"\n\nIMPORTANT: Add exactly ONE download link at the end. Do NOT write 'Download CSV' or 'using the following link' before the link. Mention the count, then add only: [Download CSV]({csv_path})"
                                    
                                    final_prompt = f"""
                                        You are a helpful assistant. Your task is ONLY to explain database query results
                                        to the user in clear, natural language.

                                        Constraints for your answer:
                                        - Do NOT mention or describe table names, column names, joins, or SQL syntax.
                                        - Do NOT reveal anything about database schema or internal implementation.
                                        - Answer ONLY for the active user_id if relevant: {user_id}.
                                        - Be concise and focus on what the numbers mean for the question.{csv_link_instruction}

                                        User Question:
                                        {user_question}

                                        SQL Query (for your reference only, do NOT explain it explicitly):
                                        {sql_query}

                                        SQL Query Results:
                                        {query_result}

                                        Now provide a short, user-friendly answer to the question based on these results.
                                        """.strip()
                                    
                                    from langchain_core.messages import HumanMessage
                                    final_messages = [HumanMessage(content=final_prompt)]
                                    response = self._invoke_llm_with_fallback(final_messages, use_tools=False, question=user_question)
                                    final_answer = response.content if hasattr(response, 'content') else str(response)
                                    logger.info(f"✅ Generated final answer from query results")
                                    return {
                                        **state,
                                        "final_answer": final_answer,
                                        "sql_query": sql_query,
                                        "query_result": query_result
                                    }
                            except Exception as exec_error:
                                logger.error(f"Error executing SQL from final message: {exec_error}")
                                logger.exception("SQL execution error in format_answer")
                    
                    final_answer = last_ai_message.content
                else:
                    logger.warning("No suitable AI message found for final answer")
                    final_answer = "I couldn't generate a response to your question."
            else:
                logger.warning("No messages available for final answer")
                final_answer = "I couldn't generate a response to your question."
        
        return {
            **state,
            "final_answer": final_answer
        }
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return results.
        
        Args:
            question: User's natural language question
            
        Returns:
            Dictionary with answer, SQL query, and results
        """
        t_request = time.perf_counter()
        logger.info(f"process=request start user_id={self.user_id} question_len={len(question)}")
        
        initial_state = {
            "question": question,
            "user_id": self.user_id,
            "messages": [],
            "examples_retrieved": False,
            "sql_query": None,
            "query_result": None,
            "final_answer": None,
            "iteration_count": 0,
            "token_usage": {"input": 0, "output": 0, "total": 0},
            "query_validated": False,
            "llm_call_history": [],
            "stage_breakdown": [],
            "csv_id": None,
            "csv_download_path": None,
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Extract query result from messages if not already set
        query_result = final_state.get("query_result")
        if not query_result:
            # Try to extract from tool messages (check all query tools, including journey tools)
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                    query_result = msg.content
                    break
        
        # Also extract SQL query from messages if not set
        sql_query = final_state.get("sql_query")
        if not sql_query:
            # Try to find SQL query in tool calls (check all query tools, including journey tools)
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call["name"]
                        if tool_name in ["execute_db_query", "count_query", "list_query"]:
                            sql_query = tool_call["args"].get("query", "")
                            break
                        elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                            # Journey tools use "sql" parameter
                            sql_query = tool_call["args"].get("sql", "")
                            break
                    if sql_query:
                        break
        
        debug_info = self._build_debug_info(final_state, question)
        elapsed_request = time.perf_counter() - t_request
        token_usage = final_state.get("token_usage", {})
        ti, to, tt = token_usage.get("input", 0), token_usage.get("output", 0), token_usage.get("total", 0)
        logger.info(f"process=request time={elapsed_request:.2f}s in={ti} out={to} total={tt}")

        return {
            "answer": final_state.get("final_answer", "No answer generated"),
            "sql_query": sql_query or final_state.get("sql_query", ""),
            "query_result": query_result or "",
            "debug": debug_info,
            "csv_id": final_state.get("csv_id"),
            "csv_download_path": final_state.get("csv_download_path"),
        }
    
    def _build_debug_info(self, state: AgentState, question: str) -> Dict[str, Any]:
        """Build comprehensive debug information including message history and token usage."""
        messages = state.get("messages", [])
        token_usage = state.get("token_usage", {"input": 0, "output": 0, "total": 0})
        llm_call_history = state.get("llm_call_history", [])
        
        # Extract tools used from message history
        tools_used = []
        tool_names_seen = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, 'name', 'unknown')
                if tool_name not in tool_names_seen:
                    tools_used.append(tool_name)
                    tool_names_seen.add(tool_name)
            elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    if tool_name not in tool_names_seen:
                        tools_used.append(tool_name)
                        tool_names_seen.add(tool_name)
        
        # Calculate token usage from llm_call_history as source of truth
        calculated_token_usage = {"input": 0, "output": 0, "total": 0}
        for call in llm_call_history:
            call_usage = call.get("token_usage", {})
            calculated_token_usage["input"] += call_usage.get("input", 0)
            calculated_token_usage["output"] += call_usage.get("output", 0)
            calculated_token_usage["total"] += call_usage.get("total", 0)
        
        # Use calculated tokens if available, otherwise fall back to state token_usage
        # This ensures totals match llm_call_history
        final_token_usage = calculated_token_usage if calculated_token_usage["total"] > 0 else token_usage
        
        # Build message history (for backward compatibility)
        message_history = []
        for i, msg in enumerate(messages):
            msg_info = {
                "index": i,
                "type": type(msg).__name__
            }
            
            if isinstance(msg, SystemMessage):
                msg_info["content"] = msg.content
                msg_info["content_length"] = len(msg.content)
            elif isinstance(msg, HumanMessage):
                msg_info["content"] = msg.content
            elif isinstance(msg, AIMessage):
                msg_info["content"] = msg.content if msg.content else None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    msg_info["tool_calls"] = []
                    for tc in msg.tool_calls:
                        tool_call_info = {
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": tc.get("args", {})
                        }
                        if tc.get("name") in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                            tool_call_info["sql_query"] = tc.get("args", {}).get("query", "")
                        msg_info["tool_calls"].append(tool_call_info)
                
                # Extract token usage from this message if available
                if hasattr(msg, "response_metadata") and msg.response_metadata:
                    usage = msg.response_metadata.get("token_usage", {})
                    if usage:
                        msg_info["token_usage"] = {
                            "input": usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0),
                            "total": usage.get("total_tokens", 0)
                        }
            elif isinstance(msg, ToolMessage):
                msg_info["name"] = getattr(msg, "name", "unknown")
                msg_info["tool_call_id"] = getattr(msg, "tool_call_id", "unknown")
                msg_info["content"] = msg.content
                msg_info["content_length"] = len(msg.content)
            
            message_history.append(msg_info)
        
        # Extract SQL query from llm_call_history if not already set
        sql_query_from_history = state.get("sql_query")
        if not sql_query_from_history:
            # Try to find SQL query in llm_call_history
            for call in llm_call_history:
                if call.get("call_type") == "agent_node":
                    tool_calls = call.get("output", {}).get("tool_calls", [])
                    for tc in tool_calls:
                        tool_name = tc.get("name")
                        if tool_name in ["execute_db_query", "count_query", "list_query"]:
                            sql_query_from_history = tc.get("sql_query", "") or tc.get("args", {}).get("query", "")
                            break
                        elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                            sql_query_from_history = tc.get("sql_query", "") or tc.get("args", {}).get("sql", "")
                            break
                    if sql_query_from_history:
                        break
        
        debug_info = {
            "question": question,
            "user_id": state.get("user_id"),
            "total_messages": len(messages),
            "message_history": message_history,
            "llm_call_history": llm_call_history,  # Detailed LLM call tracking with all steps:
            # - security_guard: Query validation (if query is blocked/allowed)
            # - agent_node: SQL query generation (includes tool_calls with SQL queries)
            # - format_answer: Human-readable answer generation
            "tools_used": tools_used,  # Ordered list of tools actually called
            "token_usage": {
                "input_tokens": final_token_usage.get("input", 0),
                "output_tokens": final_token_usage.get("output", 0),
                "total_tokens": final_token_usage.get("total", 0)
            },
            "token_usage_calculated_from_llm_calls": calculated_token_usage["total"] > 0,  # Flag indicating if totals are from llm_call_history
            "iterations": state.get("iteration_count", 0),
            "sql_query": sql_query_from_history or state.get("sql_query"),
            "query_result": state.get("query_result", "")
        }
        
        return debug_info

