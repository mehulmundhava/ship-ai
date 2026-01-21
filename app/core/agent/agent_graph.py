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
import logging
import re

# Get logger for this module
logger = logging.getLogger("ship_rag_ai")


def log_message_sequence(messages: List, step_name: str):
    """Helper function to log message sequence in a readable format."""
    print(f"\n{'='*80}")
    print(f"📋 {step_name} - Message Sequence ({len(messages)} messages)")
    print(f"{'='*80}")
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"[{i}] SystemMessage: {content_preview}")
        elif isinstance(msg, HumanMessage):
            print(f"[{i}] HumanMessage: {msg.content}")
        elif isinstance(msg, AIMessage):
            tool_calls_info = ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_info = f" (tool_calls: {[tc.get('name') for tc in msg.tool_calls]})"
            content_preview = (msg.content[:100] + "...") if msg.content and len(msg.content) > 100 else (msg.content or "[No content]")
            print(f"[{i}] AIMessage: {content_preview}{tool_calls_info}")
        elif isinstance(msg, ToolMessage):
            content_preview = (msg.content[:100] + "...") if len(msg.content) > 100 else msg.content
            print(f"[{i}] ToolMessage: name={getattr(msg, 'name', 'unknown')}, tool_call_id={getattr(msg, 'tool_call_id', 'unknown')}")
            print(f"     Content: {content_preview}")
        else:
            print(f"[{i}] {type(msg).__name__}: {str(msg)[:100]}")
    print(f"{'='*80}\n")


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
        
        # Main tools for LLM (security guard is handled separately as direct LLM call)
        # Note: execute_db_query already handles result splitting automatically
        # The specialized tools (count_query, list_query) are available but optional
        # Journey tools are for journey-related questions only
        self.tools = [
            self.get_examples_tool, 
            self.execute_db_query_tool,  # Main tool - handles both COUNT and LIST with auto-splitting
            self.count_query_tool,  # Optional: specialized for COUNT queries
            self.list_query_tool,  # Optional: specialized for LIST queries
            self.get_extra_examples_tool,  # Alias for get_few_shot_examples
            self.get_table_list_tool,
            self.get_table_structure_tool,
            self.journey_list_tool,  # For journey lists and facility breakdowns
            self.journey_count_tool  # For journey counts
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self.graph = self._build_graph()
    
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
                        logger.info(f"✅ Extracted from_facility from question: {from_facility}")
                        print(f"✅ Extracted from_facility from question: {from_facility}")
                        break
            
            if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"\n{'='*80}")
                print(f"🛠️  TOOL EXECUTION PHASE")
                print(f"{'='*80}")
                for tc in last_message.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    
                    # Inject from_facility into journey tool params if extracted from question
                    if from_facility and tool_name in ["journey_list_tool", "journey_count_tool"]:
                        if not tool_args.get("params"):
                            tool_args["params"] = {}
                        if "from_facility" not in tool_args["params"]:
                            tool_args["params"]["from_facility"] = from_facility
                            # Update the tool_call args directly (dicts are mutable)
                            tc["args"] = tool_args
                            logger.info(f"✅ Injected from_facility={from_facility} into {tool_name} params")
                            print(f"✅ Injected from_facility={from_facility} into {tool_name} params")
                    
                    print(f"📞 Calling Tool: {tool_name}")
                    if tool_name in ["execute_db_query", "count_query", "list_query"]:
                        print(f"   SQL Query: {tool_args.get('query', 'N/A')}")
                    elif tool_name in ["get_few_shot_examples", "get_extra_examples"]:
                        print(f"   Search Query: {tool_args.get('question', 'N/A')}")
                    elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                        print(f"   SQL Query: {tool_args.get('sql', 'N/A')[:100]}...")
                        if tool_args.get("params"):
                            print(f"   Params: {tool_args.get('params')}")
                print(f"{'='*80}\n")
            
            # Execute tools - this will return a state with only ToolMessages
            result = tool_node.invoke(state)
            
            # CRITICAL: Merge the preserved messages with the new ToolMessages
            # The ToolNode returns only ToolMessages, so we need to append them to the full history
            new_tool_messages = result.get("messages", [])
            
            # Combine: preserved messages (System, Human, AI) + new ToolMessages
            combined_messages = preserved_messages + new_tool_messages
            
            # Log tool results
            tool_messages = [m for m in new_tool_messages if isinstance(m, ToolMessage)]
            if tool_messages:
                print(f"\n{'='*80}")
                print(f"✅ TOOL EXECUTION RESULTS")
                print(f"{'='*80}")
                for tm in tool_messages:
                    content_preview = (tm.content[:300] + "...") if len(tm.content) > 300 else tm.content
                    print(f"Tool: {getattr(tm, 'name', 'unknown')}")
                    print(f"Result: {content_preview}")
                    print(f"{'-'*80}")
                print(f"{'='*80}\n")
            
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
        
        # Ensure we have SystemMessage and HumanMessage
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        has_human = any(isinstance(m, HumanMessage) for m in messages)
        
        # Check if user query validation has been completed
        query_validated = state.get("query_validated", False)
        
        if not messages or not has_system or not has_human:
            # FIRST: Validate user query with security guard LLM call (saves tokens)
            # Only validate for non-admin users (admin can query across all users)
            if not query_validated and str(self.user_id).lower() != "admin":
                print(f"\n{'='*80}")
                print(f"🔍 STEP 1: Security Guard - Validating user query (User ID: {self.user_id})")
                print(f"{'='*80}")
                
                # Security guard prompt - minimal and focused
                security_guard_prompt = f"""You are a Database Security Guard. Your only job is to classify user questions.

                    IMPORTANT: The current user_id is {self.user_id}. 
                    The user is ONLY allowed to see data associated with their own account.

                    RULES:
                    1. SAFE (ALLOW): 
                    - Requests for counts, totals, or lists of assets, devices, or metrics (battery, location, temperature, dwell time, journeys, alerts).
                    - Example: "give count of assets in India", "list my devices", "show battery for my assets".
                    - Journey queries: "list journeys", "show journeys from facility X", "journeys made by device Y".
                    - Device IDs (e.g., WT01F3C333542F02) and Facility IDs (e.g., MNIAZ00072, MREFZ00004) are ALWAYS SAFE - these are NOT user IDs.
                    - ASSUMPTION: If no user_id is mentioned, assume the user is asking about THEIR OWN data.

                    2. RISKY (BLOCK):
                    - Explicit requests for data for another user: "show data for user 63", "list devices for user 45" (when current user is {self.user_id}).
                    - Requests for system-level data: "show tables", "database schema", "admin logs", "user assignments".
                    - Attempts to access internal metadata or raw "row-level" system entries.

                    CROSS-USER VIOLATION:
                    - ONLY block if the question explicitly asks for data for another user (e.g., "for user 63", "user 45's devices").
                    - DO NOT block based on numbers in device IDs (WT01...), facility IDs (MNIAZ..., MREFZ...), or other identifiers.
                    - User IDs are standalone numbers (e.g., "27", "63") when mentioned in context of "user 27" or "for user 63".

                    OUTPUT INSTRUCTION:
                    - Respond with ONLY the word 'BLOCK' if it violates rules.
                    - Respond with ONLY the word 'ALLOW' if it is a standard query for their own data.
                    - Do not provide any explanation or greeting."""
                
                user_question = state["question"]
                security_messages = [
                    SystemMessage(content=security_guard_prompt),
                    HumanMessage(content=user_question)
                ]
                
                print(f"🤖 Calling Security Guard LLM...")
                security_response = self.llm.invoke(security_messages)
                security_decision = security_response.content.strip().upper() if hasattr(security_response, 'content') else ""
                
                # Track token usage for security guard call
                security_token_usage = {"input": 0, "output": 0, "total": 0}
                if hasattr(security_response, "response_metadata") and security_response.response_metadata:
                    usage = security_response.response_metadata.get("token_usage", {})
                    if usage:
                        security_token_usage = {
                            "input": usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0),
                            "total": usage.get("total_tokens", 0)
                        }
                        print(f"📊 Security Guard Token Usage: Input={security_token_usage['input']}, Output={security_token_usage['output']}, Total={security_token_usage['total']}")
                
                print(f"   Security Decision: {security_decision}")
                
                # Check if query is blocked
                if "BLOCK" in security_decision:
                    error_msg = "Sorry, I cannot provide that information."
                    print(f"❌ User query is BLOCKED by Security Guard. Stopping execution.")
                    print(f"{'='*80}\n")
                    
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
                        "messages": [HumanMessage(content=state["question"])]  # Minimal message for graph
                    }
                else:
                    print(f"✅ User query is ALLOWED by Security Guard. Proceeding with full prompt.")
                    print(f"{'='*80}\n")
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
                # Admin users skip security guard validation
                print(f"\n{'='*80}")
                print(f"🔍 STEP 1: Skipping Security Guard (Admin user)")
                print(f"{'='*80}")
                print(f"✅ Admin user - no validation needed. Proceeding with full prompt.")
                print(f"{'='*80}\n")
                state["query_validated"] = True
            
            # STEP 2: Initialize messages with FULL prompt (only if query is validated)
            # Pre-load examples in the system prompt to reduce token usage
            system_prompt = get_system_prompt(
                user_id=self.user_id,
                top_k=self.top_k,
                question=state["question"],
                vector_store_manager=self.vector_store_manager,
                preload_examples=True
            )
            base_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["question"])
            ]
            
            # Add any existing AIMessages and ToolMessages
            existing_ai = [m for m in messages if isinstance(m, AIMessage)]
            existing_tools = [m for m in messages if isinstance(m, ToolMessage)]
            
            messages = base_messages + existing_ai + existing_tools
            print(f"📝 Initialized message sequence with FULL prompt ({len(messages)} messages)")
            log_message_sequence(messages, "Initial Messages Created (After Validation)")
        else:
            # Messages are already in good shape, just validate the sequence
            print(f"📋 Processing {len(messages)} existing messages")
            
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
                        # ToolMessage without matching AIMessage - this shouldn't happen
                        # but keep it anyway to preserve tool results
                        print(f"⚠️  ToolMessage {tool_call_id} has no matching AIMessage, but keeping it")
                        valid_messages.append(msg)
            
            messages = valid_messages
            
            # Log message types
            msg_types = [type(m).__name__ for m in messages]
            print(f"   Message types: {msg_types}")
        
        # Before invoking LLM, ensure message sequence is valid
        # CRITICAL: Keep ALL messages including ToolMessages - they contain tool results!
        # The LLM needs to see the tool results to make decisions
        print(f"🔍 Validating message sequence before LLM invocation...")
        
        # Count message types
        msg_counts = {}
        for msg in messages:
            msg_type = type(msg).__name__
            msg_counts[msg_type] = msg_counts.get(msg_type, 0) + 1
        
        print(f"   Message counts: {msg_counts}")
        
        # Check if we have ToolMessages
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_messages:
            print(f"   ⚠️  Found {len(tool_messages)} ToolMessage(s) - these MUST be included so LLM can see tool results!")
            for tm in tool_messages:
                print(f"      - ToolMessage: name={getattr(tm, 'name', 'unknown')}, tool_call_id={getattr(tm, 'tool_call_id', 'unknown')}")
        
        # Filter out check_user_query_restriction tool messages from history
        # This tool is only for validation and shouldn't be included in subsequent LLM calls
        # to reduce token consumption
        filtered_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Check if this is a check_user_query_restriction tool message
                tool_name = getattr(msg, 'name', '')
                if tool_name == 'check_user_query_restriction':
                    print(f"   🔇 Filtering out check_user_query_restriction ToolMessage (not sent to LLM to save tokens)")
                    continue
            filtered_messages.append(msg)
        
        messages = filtered_messages
        
        # Invoke LLM with tools
        try:
            if not messages:
                raise ValueError("Cannot invoke LLM with empty messages array")
            
            # Debug: Log message sequence before invoking
            log_message_sequence(messages, f"Before LLM Invocation (Iteration {state.get('iteration_count', 0) + 1})")
            
            # Extract prompt details for logging
            system_msg = next((m for m in messages if isinstance(m, SystemMessage)), None)
            human_msg = next((m for m in messages if isinstance(m, HumanMessage)), None)
            if system_msg:
                print(f"📋 System Prompt Length: {len(system_msg.content)} characters")
            if human_msg:
                print(f"💬 User Question: {human_msg.content}")
            
            # Get model name in a provider-agnostic way
            model_name = getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model', None) or 'Unknown'
            print(f"🤖 Invoking LLM (model: {model_name})...")
            
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
                response = self.llm_with_tools.invoke(messages)
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                
                # Log full error details
                logger.error(f"Groq API tool call error detected")
                logger.error(f"Error Type: {error_type}")
                logger.error(f"Error Message: {error_str}")
                logger.exception("Full exception details")
                
                print(f"\n{'='*80}")
                print(f"❌ GROQ TOOL CALL ERROR DETAILS")
                print(f"{'='*80}")
                print(f"Error Type: {error_type}")
                print(f"Error Message: {error_str}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
                print(f"{'='*80}\n")
                
                # Check if it's a Groq tool call format error
                if "tool_use_failed" in error_str or "Failed to call a function" in error_str or "tool" in error_str.lower():
                    logger.warning("Groq API tool call error detected. Attempting to continue without tools...")
                    print(f"⚠️  Groq API tool call error detected. Attempting to continue without tools...")
                    # Try invoking without tools to get a text response
                    try:
                        response = self.llm.invoke(messages)
                        logger.info("Got text response from LLM (no tools)")
                        print(f"✅ Got text response from LLM (no tools)")
                    except Exception as e2:
                        logger.error(f"Error even without tools: {e2}")
                        logger.exception("Error invoking LLM without tools")
                        print(f"❌ Error even without tools: {e2}")
                        raise e
                else:
                    logger.error("Unknown error type, re-raising")
                    raise e
            
            messages.append(response)
            
            # Track token usage
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
                    print(f"📊 Token Usage: Input={call_token_usage['input']}, Output={call_token_usage['output']}, Total={call_token_usage['total']}")
            
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
            
            # Log LLM response
            if hasattr(response, "tool_calls") and response.tool_calls:
                print(f"🔧 LLM Response: Requesting {len(response.tool_calls)} tool call(s)")
                logger.info(f"LLM Response: Requesting {len(response.tool_calls)} tool call(s)")
                for tc in response.tool_calls:
                    print(f"   - Tool: {tc.get('name')}")
                    logger.info(f"Tool call: {tc.get('name')}")
                    if tc.get('name') in ['execute_db_query', 'count_query', 'list_query']:
                        sql = tc.get('args', {}).get('query', 'N/A')
                        print(f"     SQL Query: {sql}")
                        logger.info(f"SQL Query from tool call ({tc.get('name')}): {sql[:200]}...")
            else:
                content_preview = (response.content[:200] + "...") if response.content and len(response.content) > 200 else (response.content or "[No content]")
                print(f"💬 LLM Response: {content_preview}")
                logger.info(f"LLM Response (text only): {content_preview}")
                
                # Try to extract SQL from text response when tool calls fail
                extracted_sql = None
                if response.content:
                    extracted_sql = self._extract_sql_from_text(response.content)
                
                if extracted_sql:
                    logger.info(f"🔍 EXTRACTED SQL FROM TEXT RESPONSE")
                    logger.info(f"SQL Query: {extracted_sql}")
                    print(f"\n{'='*80}")
                    print(f"🔍 EXTRACTED SQL FROM TEXT RESPONSE")
                    print(f"{'='*80}")
                    print(f"SQL Query: {extracted_sql}")
                    print(f"{'='*80}\n")
                    
                    # Manually execute the query
                    try:
                        logger.info("Attempting to execute extracted SQL query")
                        print(f"🔧 Attempting to execute extracted SQL query...")
                        
                        # Use the underlying query tool instance to execute the SQL
                        query_result = self.query_tool_instance.execute(extracted_sql)
                        
                        # Create ToolMessage to maintain message flow
                        tool_message = ToolMessage(
                            content=query_result,
                            name="execute_db_query",
                            tool_call_id="manual_extraction_001"
                        )
                        messages.append(tool_message)
                        
                        # Update state
                        state["sql_query"] = extracted_sql
                        state["query_result"] = query_result
                        
                        logger.info(f"✅ Executed extracted SQL query successfully")
                        logger.info(f"Result length: {len(query_result)} characters")
                        logger.debug(f"Query result preview: {query_result[:500]}...")
                        print(f"✅ Executed extracted SQL query successfully")
                        print(f"   Result length: {len(query_result)} characters")
                    except Exception as exec_error:
                        logger.error(f"❌ Error executing extracted SQL: {exec_error}")
                        logger.exception("SQL execution error")
                        print(f"❌ Error executing extracted SQL: {exec_error}")
                        import traceback
                        traceback.print_exc()
                        # Continue with text response even if execution fails
        except Exception as e:
            print(f"❌ Error invoking LLM: {e}")
            print(f"   Messages count: {len(messages)}")
            if messages:
                print(f"   Message types: {[type(m).__name__ for m in messages]}")
                # Show detailed message info
                for i, msg in enumerate(messages):
                    if isinstance(msg, ToolMessage):
                        print(f"   [{i}] ToolMessage: tool_call_id={getattr(msg, 'tool_call_id', 'None')}, name={getattr(msg, 'name', 'None')}")
                    elif isinstance(msg, AIMessage):
                        print(f"   [{i}] AIMessage: has_tool_calls={hasattr(msg, 'tool_calls') and bool(msg.tool_calls)}")
                    else:
                        print(f"   [{i}] {type(msg).__name__}")
            raise
        
        # Extract SQL query if execute_db_query or journey tools were called
        sql_query = state.get("sql_query")
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.debug(f"Extracting SQL from {len(response.tool_calls)} tool calls")
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                if tool_name in ["execute_db_query", "count_query", "list_query"]:
                    sql_query = tool_call["args"].get("query", "")
                    logger.info(f"✅ Extracted SQL from tool call ({tool_name}): {sql_query[:100]}...")
                    break
                elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                    # Journey tools use "sql" parameter instead of "query"
                    sql_query = tool_call["args"].get("sql", "")
                    logger.info(f"✅ Extracted SQL from journey tool call ({tool_name}): {sql_query[:100]}...")
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
        """
        messages = state.get("messages", [])
        
        # Check if we have query results from any query tool (including journey tools)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                # We have query results - format answer directly without another agent call
                logger.info(f"TOOL RESULT DETECTED ({msg.name}) - Routing directly to format_answer")
                print(f"\n{'='*80}")
                print(f"✅ TOOL RESULT DETECTED ({msg.name}) - Routing directly to format_answer")
                print(f"   This saves tokens by skipping agent node with full history")
                print(f"{'='*80}\n")
                return "format"
        
        # No query results yet, continue to agent
        logger.debug("No tool results found, continuing to agent")
        return "continue"
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue (call tools) or end."""
        # Check if final_answer is already set (e.g., from early validation rejection)
        final_answer = state.get("final_answer")
        if final_answer:
            logger.info("Decision: END - Final answer already set")
            print(f"\n{'='*80}")
            print(f"✅ DECISION POINT - Final answer already set")
            print(f"{'='*80}")
            print(f"   Final answer: {final_answer[:100]}...")
            print(f"✅ Decision: END - Final answer already generated")
            return "end"
        
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        iteration_count = state.get("iteration_count", 0)
        
        logger.debug(f"Decision point - Iteration: {iteration_count}, Last message type: {type(last_message).__name__ if last_message else 'None'}")
        print(f"\n{'='*80}")
        print(f"🤔 DECISION POINT - Should Continue?")
        print(f"{'='*80}")
        print(f"   Iteration: {iteration_count}")
        print(f"   Last message type: {type(last_message).__name__ if last_message else 'None'}")
        
        # Safety check: prevent infinite loops
        if iteration_count >= 5:
            logger.warning(f"Max iterations reached ({iteration_count}), forcing end")
            print(f"⚠️  Max iterations reached ({iteration_count}), forcing end")
            return "end"
        
        # If last message has tool calls, continue to tools
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_count = len(last_message.tool_calls)
            logger.info(f"Decision: CONTINUE - Agent wants to call {tool_count} tool(s)")
            print(f"🔧 Decision: CONTINUE - Agent wants to call {tool_count} tool(s)")
            return "continue"
        
        # If we have a query result, format the answer
        query_result = state.get("query_result")
        if query_result and query_result != "":
            logger.info("Decision: END - Query result found in state")
            print(f"✅ Decision: END - Query result found")
            return "end"
        
        # Check if we have a tool message with results (from any query tool, including journey tools)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                # We have query results, format answer
                logger.info(f"Decision: END - Found {msg.name} ToolMessage")
                print(f"✅ Decision: END - Found {msg.name} ToolMessage")
                return "end"
        
        # If we have SQL query but no result yet, check if we already executed
        if state.get("sql_query"):
            # Check if we already tried to execute (check all query tools, including journey tools)
            for msg in messages:
                if isinstance(msg, ToolMessage) and msg.name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                    # Already executed, should format answer
                    logger.info(f"Decision: END - SQL query already executed via {msg.name}")
                    print(f"✅ Decision: END - SQL query already executed via {msg.name}")
                    return "end"
        
        # If last message is AIMessage without tool_calls, we have final answer
        if last_message and isinstance(last_message, AIMessage):
            has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
            if not has_tool_calls:
                logger.info("Decision: END - Final AIMessage without tool_calls")
                print(f"✅ Decision: END - Final AIMessage without tool_calls")
                return "end"
        
        # Otherwise, continue (agent might want to call tools)
        logger.debug("Decision: CONTINUE - Agent might want to call tools")
        print(f"🔄 Decision: CONTINUE - Agent might want to call tools")
        print(f"{'='*80}\n")
        return "continue"
    
    def _format_answer(self, state: AgentState) -> AgentState:
        """Format final natural language answer."""
        print(f"\n{'='*80}")
        print(f"📝 FORMATTING FINAL ANSWER")
        print(f"{'='*80}")
        
        messages = state.get("messages", [])
        log_message_sequence(messages, "All Messages Before Formatting")
        
        # Extract query result from messages if not in state
        query_result = state.get("query_result", "")
        journey_tool_used = False
        if not query_result:
            # Try to extract from tool messages (check all query tools, including journey tools)
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', '')
                    if tool_name in ["execute_db_query", "count_query", "list_query", "journey_list_tool", "journey_count_tool"]:
                        query_result = msg.content
                        if tool_name in ["journey_list_tool", "journey_count_tool"]:
                            journey_tool_used = True
                        print(f"✅ Found query result from ToolMessage ({tool_name}): {query_result[:200]}...")
                        break
        
        # Extract SQL query from messages if not in state
        sql_query = state.get("sql_query", "")
        if not sql_query:
            # Try to find SQL query in tool calls (check all query tools, including journey tools)
            logger.debug("SQL query not in state, searching in messages...")
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "")
                        if tool_name in ["execute_db_query", "count_query", "list_query"]:
                            sql_query = tool_call["args"].get("query", "")
                            logger.info(f"✅ Found SQL query from tool call ({tool_name}): {sql_query[:100]}...")
                            print(f"✅ Found SQL query from tool call ({tool_name}): {sql_query}")
                            break
                        elif tool_name in ["journey_list_tool", "journey_count_tool"]:
                            # Journey tools use "sql" parameter
                            sql_query = tool_call["args"].get("sql", "")
                            logger.info(f"✅ Found SQL query from journey tool call ({tool_name}): {sql_query[:100]}...")
                            print(f"✅ Found SQL query from journey tool call ({tool_name}): {sql_query}")
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
                            logger.info(f"✅ Extracted SQL from final message: {sql_query[:100]}...")
                            print(f"✅ Extracted SQL from final message: {sql_query}")
                            break
        
        logger.info(f"📊 Extracted SQL Query: {sql_query or 'None'}")
        logger.info(f"📊 Query Result: {query_result[:200] if query_result else 'None'}...")
        print(f"📊 Extracted SQL Query: {sql_query or 'None'}")
        print(f"📊 Query Result: {query_result[:200] if query_result else 'None'}...")
        
        # If we have query results, format them
        if query_result and not query_result.startswith("::::::"):
            logger.info("Formatting final answer from query results")
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
                        csv_download_url = f"http://localhost:3009{csv_path}"
                        logger.info(f"✅ Extracted CSV download link from JSON: {csv_download_url}")
                        print(f"✅ Extracted CSV download link from JSON: {csv_download_url}")
            except (json.JSONDecodeError, KeyError, TypeError):
                # Not JSON or no CSV link, try text format
                pass
            
            # For journey tools, reduce token usage by filtering/truncating results before format_answer
            if journey_tool_used and result_dict:
                # If there are many journeys, only include a summary in the prompt
                journies = result_dict.get('journies', [])
                total_journeys = len(journies)
                
                if total_journeys > 10:
                    # For large result sets, only include first 5 journeys in prompt
                    # The CSV link will have all the data
                    result_dict['journies'] = journies[:5]
                    result_dict['_total_journeys'] = total_journeys
                    result_dict['_showing_preview'] = True
                    query_result = json.dumps(result_dict, indent=2, default=str)
                    logger.info(f"📊 Reduced journey list from {total_journeys} to 5 for format_answer (token optimization)")
                    print(f"📊 Token optimization: Showing only 5 of {total_journeys} journeys in prompt")
            
            # If not found in JSON, try text format (for regular SQL queries)
            if not csv_download_url:
                csv_link_match = re.search(r'CSV Download Link:\s*(/download-csv/[^\s\n]+)', query_result)
                if csv_link_match:
                    csv_path = csv_link_match.group(1)
                    csv_download_url = f"http://localhost:3009{csv_path}"
                    logger.info(f"✅ Extracted CSV download link from text: {csv_download_url}")
                    print(f"✅ Extracted CSV download link from text: {csv_download_url}")

            # Build the prompt with CSV link if available
            csv_link_instruction = ""
            if csv_download_url:
                csv_link_instruction = f"\n\nIMPORTANT: The query results include a CSV download link. You MUST include this full URL in your answer: {csv_download_url}"

            final_prompt = f"""
                You are a helpful assistant. Your task is ONLY to explain database query results
                to the user in clear, natural language.

                Constraints for your answer:
                - Do NOT mention or describe table names, column names, joins, or SQL syntax.
                - Do NOT reveal anything about database schema or internal implementation.
                - Answer ONLY for the active user_id if relevant: {user_id}.
                - Be concise and focus on what the numbers mean for the question.{csv_link_instruction}{journey_instruction}

                User Question:
                {user_question}

                SQL Query (for your reference only, do NOT explain it explicitly):
                {sql_query}

                Query Results:
                {query_result}

                Now provide a short, user-friendly answer to the question based on these results.
                """.strip()
            
            logger.info("Generating final answer from query results")
            logger.debug(f"Final prompt length: {len(final_prompt)} characters")
            print(f"🤖 Generating final answer from query results...")
            print(f"   Prompt length: {len(final_prompt)} characters")
            print(f"   Using MINIMAL prompt (no system prompt, no examples, no history)")
            
            # Track what's being sent to LLM for final answer
            llm_call_history = state.get("llm_call_history", [])
            
            # Use HumanMessage format for proper LLM invocation
            from langchain_core.messages import HumanMessage
            final_messages = [HumanMessage(content=final_prompt)]
            response = self.llm.invoke(final_messages)
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            # Track token usage for this final call
            call_token_usage = {"input": 0, "output": 0, "total": 0}
            if hasattr(response, "response_metadata") and response.response_metadata:
                usage = response.response_metadata.get("token_usage", {})
                if usage:
                    call_token_usage = {
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0)
                    }
                    print(f"📊 Final Answer Token Usage: Input={call_token_usage['input']}, Output={call_token_usage['output']}, Total={call_token_usage['total']}")
            
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
            
            logger.info(f"✅ Final Answer Generated: {final_answer[:200]}...")
            print(f"✅ Final Answer Generated: {final_answer[:200]}...")
            
            return {
                **state,
                "final_answer": final_answer,
                "llm_call_history": llm_call_history,
                "token_usage": updated_token_usage
            }
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
                            print(f"✅ Extracted SQL from final message: {sql_query}")
                            
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
                                                csv_download_url = f"http://localhost:3009{csv_path}"
                                                logger.info(f"✅ Extracted CSV download link from JSON: {csv_download_url}")
                                                print(f"✅ Extracted CSV download link from JSON: {csv_download_url}")
                                    except (json.JSONDecodeError, KeyError, TypeError):
                                        pass
                                    
                                    # If not found in JSON, try text format
                                    if not csv_download_url:
                                        csv_link_match = re.search(r'CSV Download Link:\s*(/download-csv/[^\s\n]+)', query_result)
                                        if csv_link_match:
                                            csv_path = csv_link_match.group(1)
                                            csv_download_url = f"http://localhost:3009{csv_path}"
                                            logger.info(f"✅ Extracted CSV download link from text: {csv_download_url}")
                                            print(f"✅ Extracted CSV download link from text: {csv_download_url}")

                                    # Build the prompt with CSV link if available
                                    csv_link_instruction = ""
                                    if csv_download_url:
                                        csv_link_instruction = f"\n\nIMPORTANT: The query results include a CSV download link. You MUST include this full URL in your answer: {csv_download_url}"
                                    
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
                                    response = self.llm.invoke(final_messages)
                                    final_answer = response.content if hasattr(response, 'content') else str(response)
                                    logger.info(f"✅ Generated final answer from query results")
                                    print(f"✅ Final Answer: {final_answer}")
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
        
        print(f"✅ Final Answer: {final_answer}")
        print(f"{'='*80}\n")
        
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
        print(f"\n{'#'*80}")
        print(f"🚀 STARTING NEW REQUEST")
        print(f"{'#'*80}")
        print(f"📝 User Question: {question}")
        print(f"👤 User ID: {self.user_id}")
        print(f"{'#'*80}\n")
        
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
            "llm_call_history": []
        }
        
        # Run the graph
        print("🔄 Executing LangGraph workflow...\n")
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
        
        # Build debug information with message history and token usage
        debug_info = self._build_debug_info(final_state, question)
        
        # Final summary logging
        print(f"\n{'#'*80}")
        print(f"✅ REQUEST COMPLETED")
        print(f"{'#'*80}")
        print(f"📝 Question: {question}")
        print(f"💬 Answer: {final_state.get('final_answer', 'No answer generated')}")
        print(f"🔍 SQL Query: {sql_query or 'None'}")
        print(f"📊 Query Result: {query_result[:200] if query_result else 'None'}...")
        print(f"🔄 Total Iterations: {final_state.get('iteration_count', 0)}")
        token_usage = final_state.get("token_usage", {})
        print(f"💰 Token Usage: Input={token_usage.get('input', 0)}, Output={token_usage.get('output', 0)}, Total={token_usage.get('total', 0)}")
        print(f"{'#'*80}\n")
        
        return {
            "answer": final_state.get("final_answer", "No answer generated"),
            "sql_query": sql_query or final_state.get("sql_query", ""),
            "query_result": query_result or "",
            "debug": debug_info
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

