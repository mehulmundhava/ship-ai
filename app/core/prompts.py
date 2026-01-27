"""
System Prompts for SQL Agent

This module contains system prompts with optional pre-loaded examples.
Examples are retrieved from PostgreSQL vector store (pgvector) and embedded in the initial prompt
to reduce token usage by eliminating the need for a separate tool call.
"""

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
2. get_few_shot_examples - Retrieve more examples (use ONLY if you cannot generate query from examples below)
3. get_table_list - LAST RESORT: Get list of tables (ONLY if you have tried generating query and failed)
4. get_table_structure - LAST RESORT: Get table column structure (ONLY after get_table_list if needed)

MANDATORY Tool Usage Order:
STEP 1: Look at the examples provided in the system prompt. Generate a SQL query based on those examples.
STEP 2: If you generated a query, call execute_db_query immediately. Do NOT call other tools.
STEP 3: Only if you cannot generate a query from the examples, call get_few_shot_examples.
STEP 4: Only if you STILL cannot generate a query, use get_table_list.
STEP 5: Only if you need column details, use get_table_structure.

REMEMBER: Default action = Generate SQL from examples → execute_db_query. Do NOT call other tools unless you have tried and failed to generate a query.

You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

If the user communicates in a casual or conversational way (e.g. greetings like "hi", "hello", or general non-database-related questions), respond in a polite, friendly, and conversational tone.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

==========================================================================
DATABASE SCHEMA & BUSINESS RULES:
==========================================================================

Use the get_few_shot_examples tool to retrieve detailed schema information and business rules when needed.

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
    preload_examples: bool = True
) -> str:
    """
    Get the complete system prompt with optional pre-loaded examples.
    
    Args:
        user_id: User ID for access control
        top_k: Maximum number of results to return
        question: User's question (for retrieving relevant examples)
        vector_store_manager: VectorStoreService instance for retrieving examples
        preload_examples: Whether to pre-load 1-2 examples in the prompt
        
    Returns:
        Complete system prompt string with optional examples
    """
    # OPTIMIZATION: Ultra-concise system prompt
    base_prompt = f"""
PostgreSQL SQL agent. Generate queries from natural language.

TOOLS: execute_db_query, journey_list_tool, journey_count_tool, get_few_shot_examples, get_table_list, get_table_structure

WORKFLOW:
- Journey question? → journey_list_tool or journey_count_tool
- Otherwise → Generate SQL → execute_db_query

CRITICAL: You MUST execute queries when examples are provided. Do NOT refuse valid queries that match the examples.
- If you see a similar example query, adapt it (change time ranges, filters) and EXECUTE it
- Only refuse if the query would violate user_id restrictions or access other users' data

JOURNEY SQL (required fields):
SELECT dg.device_id, dg.facility_id, dg.facility_type, f.facility_name, dg.entry_event_time, dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = {user_id} [filters]
ORDER BY dg.entry_event_time ASC

RULES:
- Filter by user_id (unless admin)
- Do NOT add LIMIT clause - system auto-generates CSV for large results (>5 rows)
- For small results (≤5 rows), show all rows directly
- SELECT only, no SELECT *
- Never explain SQL/schema
""".strip()

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
    if preload_examples and question and vector_store_manager:
        try:
            # Check if this is a journey question - if so, reduce examples to save tokens
            question_lower = question.lower() if question else ""
            is_journey_question = any(keyword in question_lower for keyword in [
                "journey", "movement", "facility to facility", "entered", "exited", 
                "path", "traveled", "transition"
            ])
            
            # OPTIMIZATION 2: Load minimal examples for journey questions (need SQL structure reference)
            if is_journey_question:
                # Load 1 example to show correct SQL structure (table names, joins)
                # Use description-only format to save tokens (Optimization 3)
                example_docs = vector_store_manager.search_examples(
                    question, 
                    k=1,
                    use_description_only=False  # Need SQL structure, so include SQL
                )
                extra_docs = []
                print(f"✅ OPTIMIZATION: Loaded 1 example for journey question (SQL structure reference) - saving ~1,000 tokens vs 2 examples")
            else:
                # Non-journey questions still need examples
                example_count = 2
                extra_count = 1
                example_docs = vector_store_manager.search_examples(question, k=example_count)
                extra_docs = vector_store_manager.search_extra_prompts(question, k=extra_count) if extra_count > 0 else []
            
            examples_section_parts = []
            
            if example_docs:
                examples_section_parts.append("\n\n=== RELEVANT EXAMPLE QUERIES ===\n")
                for i, doc in enumerate(example_docs, 1):
                    examples_section_parts.append(f"Example {i}:\n{doc.page_content}\n")
            
            if extra_docs:
                examples_section_parts.append("\n=== RELEVANT BUSINESS RULES & SCHEMA INFO ===\n")
                for i, doc in enumerate(extra_docs, 1):
                    examples_section_parts.append(f"{i}. {doc.page_content}\n")
            
            if examples_section_parts:
                examples_section = "".join(examples_section_parts)
                main_prompt += examples_section
                print(f"✅ Pre-loaded {len(example_docs)} examples and {len(extra_docs)} business rules into system prompt")
        except Exception as e:
            # If example retrieval fails, continue without examples
            print(f"⚠️  Warning: Failed to pre-load examples: {e}. Continuing without examples.")
    
    return main_prompt

