"""
LangGraph Agent Implementation

This module implements the agentic RAG workflow using LangGraph.
The agent can conditionally call tools to retrieve examples or execute queries.
"""

import json
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.utilities.sql_database import SQLDatabase
from agent_tools import create_get_few_shot_examples_tool, create_execute_db_query_tool
from prompts import get_system_prompt
from vector_store import VectorStoreManager


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


class SQLAgentGraph:
    """
    LangGraph-based SQL agent with RAG capabilities.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        db: SQLDatabase,
        vector_store_manager: VectorStoreManager,
        user_id: Optional[str] = None,
        top_k: int = 20
    ):
        """
        Initialize the SQL agent graph.
        
        Args:
            llm: Language model instance
            db: SQLDatabase instance
            vector_store_manager: VectorStoreManager instance
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
        
        self.tools = [self.get_examples_tool, self.execute_db_query_tool]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self.graph = self._build_graph()
    
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
            
            # CRITICAL: Preserve the full message history before tool execution
            # The ToolNode will only return ToolMessages, so we need to keep everything
            preserved_messages = messages.copy()
            
            if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"\n{'='*80}")
                print(f"🛠️  TOOL EXECUTION PHASE")
                print(f"{'='*80}")
                for tc in last_message.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    print(f"📞 Calling Tool: {tool_name}")
                    if tool_name == "execute_db_query":
                        print(f"   SQL Query: {tool_args.get('query', 'N/A')}")
                    elif tool_name == "get_few_shot_examples":
                        print(f"   Search Query: {tool_args.get('question', 'N/A')}")
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
        
        # Add edges
        workflow.add_edge("tools", "agent")  # After tools, go back to agent
        workflow.add_edge("format_answer", END)
        
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Agent node that decides what to do next."""
        messages = state.get("messages", [])
        
        # Ensure we have SystemMessage and HumanMessage
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        has_human = any(isinstance(m, HumanMessage) for m in messages)
        
        if not messages or not has_system or not has_human:
            # Initialize messages if empty or missing required components
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
            print(f"📝 Initialized message sequence ({len(messages)} messages)")
            log_message_sequence(messages, "Initial Messages Created")
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
        
        # Keep ALL messages - don't filter out ToolMessages
        # The OpenAI API will validate the sequence, but we need to preserve tool results
        # If there's an issue, it will be caught by the API and we'll handle it
        messages = messages
        
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
            
            print(f"🤖 Invoking LLM (model: {self.llm.model_name})...")
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Track token usage
            token_usage = state.get("token_usage", {"input": 0, "output": 0, "total": 0})
            if hasattr(response, "response_metadata") and response.response_metadata:
                usage = response.response_metadata.get("token_usage", {})
                if usage:
                    token_usage["input"] = token_usage.get("input", 0) + usage.get("prompt_tokens", 0)
                    token_usage["output"] = token_usage.get("output", 0) + usage.get("completion_tokens", 0)
                    token_usage["total"] = token_usage.get("total", 0) + usage.get("total_tokens", 0)
                    print(f"📊 Token Usage: Input={usage.get('prompt_tokens', 0)}, Output={usage.get('completion_tokens', 0)}, Total={usage.get('total_tokens', 0)}")
            
            # Log LLM response
            if hasattr(response, "tool_calls") and response.tool_calls:
                print(f"🔧 LLM Response: Requesting {len(response.tool_calls)} tool call(s)")
                for tc in response.tool_calls:
                    print(f"   - Tool: {tc.get('name')}")
                    if tc.get('name') == 'execute_db_query':
                        print(f"     SQL Query: {tc.get('args', {}).get('query', 'N/A')}")
            else:
                content_preview = (response.content[:200] + "...") if response.content and len(response.content) > 200 else (response.content or "[No content]")
                print(f"💬 LLM Response: {content_preview}")
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
        
        # Extract SQL query if execute_db_query was called
        sql_query = state.get("sql_query")
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "execute_db_query":
                    sql_query = tool_call["args"]["query"]
                    break
        
        # Check if we got query results from previous tool execution
        query_result = state.get("query_result")
        if not query_result:
            # Look for tool messages with query results
            for msg in messages:
                if isinstance(msg, ToolMessage) and msg.name == "execute_db_query":
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
            "token_usage": current_token_usage
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue (call tools) or end."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        iteration_count = state.get("iteration_count", 0)
        
        print(f"\n{'='*80}")
        print(f"🤔 DECISION POINT - Should Continue?")
        print(f"{'='*80}")
        print(f"   Iteration: {iteration_count}")
        print(f"   Last message type: {type(last_message).__name__ if last_message else 'None'}")
        
        # Safety check: prevent infinite loops
        if iteration_count >= 3:
            print(f"⚠️  Max iterations reached ({iteration_count}), forcing end")
            return "end"
        
        # If last message has tool calls, continue to tools
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"🔧 Decision: CONTINUE - Agent wants to call {len(last_message.tool_calls)} tool(s)")
            return "continue"
        
        # If we have a query result, format the answer
        query_result = state.get("query_result")
        if query_result and query_result != "":
            print(f"✅ Decision: END - Query result found")
            return "end"
        
        # Check if we have a tool message with results (from execute_db_query)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name == "execute_db_query":
                # We have query results, format answer
                print(f"✅ Decision: END - Found execute_db_query ToolMessage")
                return "end"
        
        # If we have SQL query but no result yet, check if we already executed
        if state.get("sql_query"):
            # Check if we already tried to execute
            for msg in messages:
                if isinstance(msg, ToolMessage) and msg.name == "execute_db_query":
                    # Already executed, should format answer
                    print(f"✅ Decision: END - SQL query already executed")
                    return "end"
        
        # If last message is AIMessage without tool_calls, we have final answer
        if last_message and isinstance(last_message, AIMessage):
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                print(f"✅ Decision: END - Final AIMessage without tool_calls")
                return "end"
        
        # Otherwise, continue (agent might want to call tools)
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
        if not query_result:
            # Try to extract from tool messages
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage) and msg.name == "execute_db_query":
                    query_result = msg.content
                    print(f"✅ Found query result from ToolMessage: {query_result[:200]}...")
                    break
        
        # Extract SQL query from messages if not in state
        sql_query = state.get("sql_query", "")
        if not sql_query:
            # Try to find SQL query in tool calls
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        if tool_call["name"] == "execute_db_query":
                            sql_query = tool_call["args"].get("query", "")
                            print(f"✅ Found SQL query from tool call: {sql_query}")
                            break
                    if sql_query:
                        break
        
        print(f"📊 Extracted SQL Query: {sql_query or 'None'}")
        print(f"📊 Query Result: {query_result[:200] if query_result else 'None'}...")
        
        # If we have query results, format them
        if query_result and not query_result.startswith("::::::"):
            # Generate final answer from results using a MINIMAL, one-shot prompt.
            # IMPORTANT: We DO NOT send the full message history, system prompt,
            # or retrieved examples again here. This keeps token usage low for
            # the final "explain the result" step.
            user_question = state.get("question", "")
            user_id = state.get("user_id", "")

            final_prompt = f"""
                You are a helpful assistant. Your task is ONLY to explain database query results
                to the user in clear, natural language.

                Constraints for your answer:
                - Do NOT mention or describe table names, column names, joins, or SQL syntax.
                - Do NOT reveal anything about database schema or internal implementation.
                - Answer ONLY for the active user_id if relevant: {user_id}.
                - Be concise and focus on what the numbers mean for the question.

                User Question:
                {user_question}

                SQL Query (for your reference only, do NOT explain it explicitly):
                {sql_query}

                SQL Query Results:
                {query_result}

                Now provide a short, user-friendly answer to the question based on these results.
                """.strip()
            
            print(f"🤖 Generating final answer from query results...")
            print(f"   Prompt length: {len(final_prompt)} characters")
            response = self.llm.invoke(final_prompt)
            final_answer = response.content if hasattr(response, 'content') else str(response)
            print(f"✅ Final Answer Generated: {final_answer[:200]}...")
        elif query_result and query_result.startswith("::::::"):
            # Empty result
            final_answer = f"Based on the query, there are no results matching your criteria for the question: {state['question']}"
        else:
            # No query executed - use last message
            if messages:
                last_ai_message = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
                        last_ai_message = msg
                        break
                if last_ai_message:
                    final_answer = last_ai_message.content
                else:
                    final_answer = "I couldn't generate a response to your question."
            else:
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
            "token_usage": {"input": 0, "output": 0, "total": 0}
        }
        
        # Run the graph
        print("🔄 Executing LangGraph workflow...\n")
        final_state = self.graph.invoke(initial_state)
        
        # Extract query result from messages if not already set
        query_result = final_state.get("query_result")
        if not query_result:
            # Try to extract from tool messages
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, ToolMessage) and msg.name == "execute_db_query":
                    query_result = msg.content
                    break
        
        # Also extract SQL query from messages if not set
        sql_query = final_state.get("sql_query")
        if not sql_query:
            # Try to find SQL query in tool calls
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        if tool_call["name"] == "execute_db_query":
                            sql_query = tool_call["args"].get("query", "")
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
        
        # Build message history
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
                        if tc.get("name") == "execute_db_query":
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
        
        debug_info = {
            "question": question,
            "user_id": state.get("user_id"),
            "total_messages": len(messages),
            "message_history": message_history,
            "token_usage": {
                "input_tokens": token_usage.get("input", 0),
                "output_tokens": token_usage.get("output", 0),
                "total_tokens": token_usage.get("total", 0)
            },
            "iterations": state.get("iteration_count", 0),
            "sql_query": state.get("sql_query"),
            "query_result": state.get("query_result", "")
        }
        
        return debug_info

