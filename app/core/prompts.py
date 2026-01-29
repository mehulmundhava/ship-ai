"""
System Prompts for SQL Agent

This module contains system prompts with optional pre-loaded examples.
Examples are retrieved from PostgreSQL vector store (pgvector) and embedded in the initial prompt
to reduce token usage by eliminating the need for a separate tool call.
"""
import logging
from typing import Optional, List

logger = logging.getLogger("ship_rag_ai")

# Security and access control instructions
STRICT_INSTRUCTION = """
The user_id for this request is: {user_id}

STRICT DATA ACCESS & SECURITY RULES (MANDATORY FOR EVERY QUERY):

- UNDER NO CIRCUMSTANCES may you provide, query, or reference data for any user other than the one specified by user_id.
- You MUST apply user_id as a strict filter condition in every query and every answer. DO NOT answer, reference, or reveal any information not tied directly to this user_id.
- If you are asked a question that is not relevant to the given user_id, or that would return data for another user, you must refuse and respond only with: "Sorry, I cannot provide that information."
- NEVER provide or discuss database schema, internal table structure, metadata, or implementation details of the database or system. If such a question is asked, you must respond only with: "Sorry, I cannot provide that information."
- You must enforce these rules for EVERY input, EVERY question, and EVERY query. There are NO EXCEPTIONS.
"""

# Concise system prompt (examples removed - retrieved via RAG)
CONCISE_SQL_PREFIX = """
You are an agent designed to interact with a PostgreSQL database.
Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

CRITICAL: Your FIRST action should be to GENERATE A SQL QUERY using the examples provided in the system prompt.
Do NOT call any tools until you have attempted to generate a SQL query from the examples.

You have access to tools for interacting with the database:
1. execute_db_query - Execute SQL queries against PostgreSQL database (use AFTER you generate a query)
2. get_table_list - LAST RESORT: Get list of tables (ONLY if you have tried generating query and failed)
3. get_table_structure - LAST RESORT: Get table column structure (ONLY after get_table_list if needed)

MANDATORY Tool Usage Order:
STEP 1: Look at the examples provided in the system prompt (from ai_vector_examples). Generate a SQL query based on those examples.
STEP 2: If you generated a query, call execute_db_query immediately. Do NOT call other tools.
STEP 3: Only if you cannot generate a query from the examples, use get_table_list.
STEP 4: Only if you need column details, use get_table_structure.

REMEMBER: Default action = Generate SQL from examples → execute_db_query. Do NOT call other tools unless you have tried and failed to generate a query.

You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

If the user communicates in a casual or conversational way (e.g. greetings like "hi", "hello", or general non-database-related questions), respond in a polite, friendly, and conversational tone.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

==========================================================================
DATABASE SCHEMA & BUSINESS RULES:
==========================================================================

Key points to remember:
- Always join user_device_assignment (ud) table and add ud.user_id where condition when user_id is not "admin"
- Use latest_incoming_message_id, latest_sensor_id, latest_shock_id, latest_free_fall_id from device_details_table to get current data
- Facility types: M (manufacturer), R (retailer), U, D, etc.
- Journey = device movement from one facility_id to another
- Dwell time is in seconds (1 day = 86400 seconds)

JOURNEY CALCULATION RULES (CRITICAL):
- Journey calculations MUST be done in Python, NOT in SQL
- SQL is ONLY for fetching raw geofencing rows from device_geofencings table
- For journey questions, use journey_list_tool or journey_count_tool
- These tools execute SQL to fetch rows, then run Python algorithm to calculate journeys
- Journey time must be >= 4 hours (14400 seconds) between different facilities
- For same facility (A -> A), minimum time is 4 hours + extraJourneyTimeLimit (if provided)
- Journey key format: "facilityA||facilityB" (e.g., "M||R" for manufacturer to retailer)

==========================================================================
"""


def get_system_prompt(
    user_id: str,
    top_k: int = 20,
    question: str = None,
    vector_store_manager = None,
    preload_examples: bool = True,
    is_journey: bool = False,
    precomputed_embedding: Optional[List[float]] = None,
) -> str:
    """
    Get the complete system prompt with optional pre-loaded examples.
    
    Args:
        user_id: User ID for access control
        top_k: Maximum number of results to return
        question: User's question (for retrieving relevant examples)
        vector_store_manager: VectorStoreService instance for retrieving examples
        preload_examples: Whether to pre-load 1-2 examples in the prompt
        is_journey: Whether the question is about journeys/movement
        precomputed_embedding: Optional embedding from 80% path miss to avoid re-embedding (saves ~450ms)
        
    Returns:
        Complete system prompt string with optional examples
    """
    # OPTIMIZATION: Ultra-concise system prompt
    # Removed fallback tools (get_table_list, get_table_structure) and redundant query tools (count_query, list_query)
    # to save tokens. The agent handles structure errors automatically via auto-retry.
    
    # Base tools available to everyone (examples from ai_vector_examples only, pre-loaded in prompt)
    tools_list = "execute_db_query, get_table_list, get_table_structure"
    
    # Workflow description
    if is_journey:
        tools_list += ", journey_list_tool, journey_count_tool"
        workflow_desc = "Journey question? → journey_list_tool or journey_count_tool"
    else:
        workflow_desc = "Generate SQL → execute_db_query"
        
    base_prompt = f"""
PostgreSQL SQL agent. Generate queries from natural language.

TOOLS: {tools_list}

WORKFLOW:
- {workflow_desc}

CRITICAL: You MUST execute queries when examples are provided. Do NOT refuse valid queries that match the examples.
- If you see a similar example query, adapt it (change time ranges, filters) and EXECUTE it
- Only refuse if the query would violate user_id restrictions or access other users' data
""".strip()

    # Only include Journey SQL template for journey questions to save tokens
    journey_sql_block = f"""
JOURNEY SQL (required fields):
SELECT dg.device_id, dg.facility_id, dg.facility_type, f.facility_name, dg.entry_event_time, dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = {user_id} [filters]
ORDER BY dg.entry_event_time ASC
""".strip()

    # Rules block
    rules_block = """
RULES:
- Filter by user_id (unless admin)
- Do NOT add LIMIT clause - system auto-generates CSV for large results (>5 rows)
- For small results (≤5 rows), show all rows directly
- SELECT only, no SELECT *
- Never explain SQL/schema
""".strip()

    # Combine blocks based on context
    if is_journey:
        base_prompt = f"{base_prompt}\n\n{journey_sql_block}\n\n{rules_block}"
    else:
        base_prompt = f"{base_prompt}\n\n{rules_block}"

    admin_prompt = f"""
ADMIN MODE: No user_id filtering required. Query across all users.
""".strip()

    user_prompt = f"""
USER MODE: user_id = {user_id}
- ALWAYS filter by ud.user_id = '{user_id}'
- ALWAYS join user_device_assignment (ud)
- Aggregations, GROUP BY, COUNT, SUM, etc. are ALLOWED for this user_id's data
- Time ranges (days, months, years) are ALLOWED - adapt examples by changing INTERVAL values
- Multiple visits, repeated facilities, patterns are ALLOWED for this user_id
- ONLY refuse if query would access OTHER users' data (user_id != {user_id})
- Follow the example queries provided - adapt them to match the question's time range
- Never explain SQL/schema in answers
""".strip()

    # Build the main prompt
    if user_id and str(user_id).lower() == "admin":
        main_prompt = "\n\n".join([base_prompt, admin_prompt])
    else:
        main_prompt = "\n\n".join([base_prompt, user_prompt])
    
    # Pre-load examples if requested and parameters are provided
    if preload_examples and (question or precomputed_embedding is not None) and vector_store_manager:
        try:
            # Check if this is a journey question
            is_journey_question = is_journey
            if not is_journey_question and question:
                question_lower = question.lower()
                is_journey_question = any(keyword in question_lower for keyword in [
                    "journey", "movement", "facility to facility", "entered", "exited",
                    "path", "traveled", "transition"
                ])

            # Training data from ai_vector_examples only (no ai_vector_extra_prompts)
            if precomputed_embedding is not None:
                logger.info("get_system_prompt: using precomputed_embedding (no re-embed)")
                if is_journey_question:
                    example_docs = vector_store_manager.search_examples_with_embedding(
                        precomputed_embedding, k=1, use_description_only=False
                    )
                else:
                    example_count = 3
                    example_docs = vector_store_manager.search_examples_with_embedding(
                        precomputed_embedding, k=example_count
                    )
            else:
                if is_journey_question:
                    example_docs = vector_store_manager.search_examples(
                        question, k=1, use_description_only=False
                    )
                else:
                    example_count = 3
                    example_docs = vector_store_manager.search_examples(question, k=example_count)

            examples_section_parts = []
            if example_docs:
                examples_section_parts.append("\n\n=== RELEVANT EXAMPLE QUERIES (ai_vector_examples) ===\n")
                for i, doc in enumerate(example_docs, 1):
                    examples_section_parts.append(f"Example {i}:\n{doc.page_content}\n")
            
            if examples_section_parts:
                examples_section = "".join(examples_section_parts)
                main_prompt += examples_section
        except Exception as e:
            pass  # Continue without examples
    
    return main_prompt

