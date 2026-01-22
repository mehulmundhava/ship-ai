"""
LangGraph Agent Tools

This module defines tools that the agent can use:
1. get_few_shot_examples - Retrieves similar examples from PostgreSQL vector store (pgvector)
2. execute_db_query - Executes SQL queries against PostgreSQL
3. get_table_list - Retrieves list of tables with descriptions and important fields
4. get_table_structure - Retrieves full column structure for specified tables
"""

from typing import Optional, Dict, Any, Sequence, Tuple, List
import re
import json
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy.engine import Result
from sqlalchemy import text
from app.config.table_metadata import TABLE_METADATA
from app.utils.csv_generator import format_result_with_csv, generate_csv_from_result, format_journey_list_with_csv
from app.core.journey_calculator import calculate_journey_counts, calculate_journey_list

# List of sensitive tables that should not be queried directly for raw data
RESTRICTED_TABLES = [
    'admin',
    'user_device_assignment',
    'users',  # if exists
    'user',   # if exists
]

def _is_restricted_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a SQL query is trying to directly access restricted sensitive tables.
    
    A query is restricted if it directly selects from a restricted table in the main FROM clause.
    Queries that use restricted tables only in JOINs are allowed (for filtering/access control).
    
    Args:
        query: SQL query string
        
    Returns:
        Tuple of (is_restricted: bool, restricted_table: Optional[str])
    """
    # Remove comments and normalize whitespace
    query_clean = re.sub(r'--.*?$', '', query, flags=re.MULTILINE)
    query_clean = re.sub(r'/\*.*?\*/', '', query_clean, flags=re.DOTALL)
    query_clean = ' '.join(query_clean.split())
    query_upper = query_clean.upper()
    
    # Check if query starts with SELECT (read-only)
    if not query_upper.strip().startswith('SELECT'):
        return False, None
    
    # Check for direct FROM clause with restricted tables
    # Pattern: FROM restricted_table (not in JOIN)
    for table in RESTRICTED_TABLES:
        # Find all FROM occurrences
        from_pattern = rf'\bFROM\s+{re.escape(table)}\b'
        from_matches = list(re.finditer(from_pattern, query_clean, re.IGNORECASE))
        
        for from_match in from_matches:
            # Check what comes after the table name
            after_table = query_clean[from_match.end():from_match.end()+50]
            # If followed by JOIN, it's part of a JOIN clause (allowed)
            # If followed by WHERE, GROUP, ORDER, LIMIT, OFFSET, or end, it's the main table (restricted)
            if re.match(r'^\s+(WHERE|GROUP|ORDER|LIMIT|OFFSET|$)', after_table, re.IGNORECASE):
                return True, table
            # Also check for comma-separated tables (FROM table1, table2)
            if re.match(r'^\s*,', after_table):
                return True, table
    
    return False, None


def create_get_few_shot_examples_tool(vector_store_manager):
    """
    Create the get_few_shot_examples tool function.
    
    Args:
        vector_store_manager: VectorStoreService instance
        
    Returns:
        Tool function for retrieving examples
    """
    @tool
    def get_few_shot_examples(question: str) -> str:
        """
        Retrieve additional similar example queries and business rules from the knowledge base.
        
        NOTE: You already have 1-2 relevant examples pre-loaded in your system prompt.
        Use this tool ONLY when:
        - You need MORE examples beyond what's already provided
        - The pre-loaded examples don't match your specific use case
        - You need additional business rules or schema information
        
        Args:
            question: The user's question to find similar examples for
            
        Returns:
            A formatted string containing similar examples and relevant context
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: get_few_shot_examples")
        print(f"{'='*80}")
        print(f"   Input Question: {question}")
        
        # Search for similar examples
        example_docs = vector_store_manager.search_examples(question, k=3)
        
        # Search for relevant extra prompt data
        extra_docs = vector_store_manager.search_extra_prompts(question, k=2)
        
        result_parts = []
        
        if example_docs:
            result_parts.append("=== SIMILAR EXAMPLE QUERIES ===\n")
            for i, doc in enumerate(example_docs, 1):
                result_parts.append(f"Example {i}:\n{doc.page_content}\n")
        
        if extra_docs:
            result_parts.append("\n=== RELEVANT BUSINESS RULES & SCHEMA INFO ===\n")
            for i, doc in enumerate(extra_docs, 1):
                result_parts.append(f"{i}. {doc.page_content}\n")
        
        if not result_parts:
            print(f"   Result: No similar examples found")
            print(f"{'='*80}\n")
            return "No similar examples found. Proceed with your knowledge of the database schema."
        
        result = "\n".join(result_parts)
        print(f"   Result: Found {len(example_docs)} examples and {len(extra_docs)} extra prompts")
        print(f"   Output length: {len(result)} characters")
        print(f"{'='*80}\n")
        
        return result
    
    return get_few_shot_examples


def create_count_query_tool(db: SQLDatabase):
    """
    Create the count_query tool function for COUNT/aggregation queries.
    
    Args:
        db: SQLDatabase instance
        
    Returns:
        Tool function for executing COUNT queries
    """
    query_tool = QuerySQLDatabaseTool(db)
    
    @tool
    def count_query(query: str) -> str:
        """
        Execute COUNT or aggregation queries and return only the count/aggregation result.
        
        Use this tool when the user asks for:
        - Counts (e.g., "how many devices", "count of assets")
        - Totals (e.g., "total temperature", "sum of battery")
        - Aggregations (e.g., "average temperature", "maximum battery")
        
        This tool is optimized for queries that return a single aggregated value.
        
        Args:
            query: SQL query that returns a count or aggregation (should use COUNT, SUM, AVG, MAX, MIN)
            
        Returns:
            Count or aggregation result as string
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: count_query")
        print(f"{'='*80}")
        print(f"   SQL Query: {query}")
        
        # Execute query
        output = query_tool.db.run_no_throw(query, include_columns=True)
        
        if not output:
            print(f"   Result: No rows returned")
            print(f"{'='*80}\n")
            return "0"
        
        result = str(output)
        print(f"   Count Result: {result}")
        print(f"{'='*80}\n")
        
        return result
    
    return count_query


def create_list_query_tool(db: SQLDatabase):
    """
    Create the list_query tool function for LIST queries.
    
    Args:
        db: SQLDatabase instance
        
    Returns:
        Tool function for executing LIST queries
    """
    query_tool = QuerySQLDatabaseTool(db)
    
    @tool
    def list_query(query: str, limit: int = 5) -> str:
        """
        Execute LIST queries and return first N rows only.
        
        Use this tool when the user asks for:
        - Lists of items (e.g., "list devices", "show assets", "get all facilities")
        - Multiple rows of data
        
        IMPORTANT: This tool automatically limits results to the first 5 rows by default.
        If the query returns more than 5 rows, it will:
        1. Return the count of total rows
        2. Show the first 5 rows
        3. Provide a CSV download link for the full results
        
        Args:
            query: SQL query that returns a list of rows
            limit: Maximum number of rows to return in preview (default: 5)
            
        Returns:
            First N rows as formatted string, with count and CSV link if > limit rows
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: list_query")
        print(f"{'='*80}")
        print(f"   SQL Query: {query}")
        print(f"   Limit: {limit}")
        
        # Ensure query has LIMIT clause (add if not present)
        query_upper = query.upper().strip()
        if 'LIMIT' not in query_upper:
            # Add LIMIT to get all rows (we'll handle splitting in result processing)
            # But first, let's execute without modifying to preserve original query
            pass
        
        # Execute query
        output = query_tool.db.run_no_throw(query, include_columns=True)
        
        if not output:
            print(f"   Result: No rows returned")
            print(f"{'='*80}\n")
            return ":::::: Query execution has returned 0 rows. Return final answer accordingly. ::::::"
        
        result = str(output)
        
        # Format result with CSV if needed
        formatted_result = format_result_with_csv(result, max_preview_rows=limit)
        
        print(f"   Formatted result with splitting logic")
        print(f"{'='*80}\n")
        
        return formatted_result
    
    return list_query


def create_get_extra_examples_tool(vector_store_manager):
    """
    Create the get_extra_examples tool function (alias for get_few_shot_examples).
    
    Args:
        vector_store_manager: VectorStoreService instance
        
    Returns:
        Tool function for retrieving additional examples
    """
    @tool
    def get_extra_examples(question: str) -> str:
        """
        Retrieve additional similar example queries beyond the 2 pre-loaded examples.
        
        NOTE: You already have 1-2 relevant examples pre-loaded in your system prompt.
        Use this tool ONLY when:
        - You need MORE examples beyond what's already provided
        - The pre-loaded examples don't match your specific use case
        - You need additional business rules or schema information
        
        This is an alias for get_few_shot_examples for clarity.
        
        Args:
            question: The user's question to find similar examples for
            
        Returns:
            A formatted string containing similar examples and relevant context
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: get_extra_examples")
        print(f"{'='*80}")
        print(f"   Input Question: {question}")
        
        # Search for similar examples (same logic as get_few_shot_examples)
        example_docs = vector_store_manager.search_examples(question, k=3)
        extra_docs = vector_store_manager.search_extra_prompts(question, k=2)
        
        result_parts = []
        
        if example_docs:
            result_parts.append("=== SIMILAR EXAMPLE QUERIES ===\n")
            for i, doc in enumerate(example_docs, 1):
                result_parts.append(f"Example {i}:\n{doc.page_content}\n")
        
        if extra_docs:
            result_parts.append("\n=== RELEVANT BUSINESS RULES & SCHEMA INFO ===\n")
            for i, doc in enumerate(extra_docs, 1):
                result_parts.append(f"{i}. {doc.page_content}\n")
        
        if not result_parts:
            print(f"   Result: No similar examples found")
            print(f"{'='*80}\n")
            return "No similar examples found. Proceed with your knowledge of the database schema."
        
        result = "\n".join(result_parts)
        print(f"   Result: Found {len(example_docs)} examples and {len(extra_docs)} extra prompts")
        print(f"   Output length: {len(result)} characters")
        print(f"{'='*80}\n")
        
        return result
    
    return get_extra_examples


class QuerySQLDatabaseTool:
    """
    Tool for executing SQL queries against PostgreSQL database.
    """
    
    def __init__(self, db: SQLDatabase):
        self.db = db
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect if query is COUNT, LIST, or other type.
        
        Args:
            query: SQL query string
            
        Returns:
            'count', 'list', or 'other'
        """
        query_upper = query.upper().strip()
        
        # Check for COUNT queries
        if re.search(r'\bCOUNT\s*\(', query_upper):
            return 'count'
        
        # Check for aggregation queries (SUM, AVG, MAX, MIN with GROUP BY or without)
        if re.search(r'\b(SUM|AVG|MAX|MIN)\s*\(', query_upper):
            # If it's just aggregation without GROUP BY, treat as count-like
            if 'GROUP BY' not in query_upper:
                return 'count'
        
        # Check for SELECT queries (list queries)
        if query_upper.startswith('SELECT'):
            return 'list'
        
        return 'other'
    
    def execute(self, query: str) -> str:
        """
        Execute a SQL query and return results.
        Automatically handles result splitting for large lists.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results as string, or error message if query fails
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: execute_db_query")
        print(f"{'='*80}")
        print(f"   SQL Query: {query}")
        
        # Detect query type
        query_type = self._detect_query_type(query)
        print(f"   Detected Query Type: {query_type}")
        
        # Execute query without throwing exceptions
        output = self.db.run_no_throw(query, include_columns=True)
        
        # Check if query returned no rows
        if not output:
            print(f"   Result: No rows returned")
            print(f"{'='*80}\n")
            return ":::::: Query execution has returned 0 rows. Return final answer accordingly. ::::::"
        
        result = str(output)
        
        # Handle result splitting for LIST queries
        if query_type == 'list':
            # Check if result is a Python list/dict format (from SQLDatabase)
            import ast
            
            row_count = 0
            is_dict_format = False
            original_result = result
            
            # Try to parse as Python list/dict
            try:
                # Check if it looks like a Python list/dict representation
                if result.strip().startswith('[') and ('{' in result or '[' in result):
                    # Try to parse as JSON-like structure
                    parsed = ast.literal_eval(result)
                    if isinstance(parsed, list):
                        row_count = len(parsed)
                        is_dict_format = True
                        # Convert to formatted string for CSV generation
                        if parsed and isinstance(parsed[0], dict):
                            # Convert list of dicts to pipe-separated format
                            headers = list(parsed[0].keys())
                            lines = [' | '.join(headers)]
                            for row in parsed:
                                values = [str(row.get(h, '')) for h in headers]
                                lines.append(' | '.join(values))
                            result = '\n'.join(lines)
                            print(f"   Converted Python dict format to pipe-separated format")
            except (ValueError, SyntaxError, AttributeError):
                # Not a Python list format, parse as string
                pass
            
            # If not dict format, parse as string
            if not is_dict_format:
                lines = result.strip().split('\n')
                data_rows = [line for line in lines[1:] if line.strip()] if len(lines) > 1 else []
                row_count = len(data_rows)
            else:
                # Count from the converted format
                lines = result.strip().split('\n')
                row_count = len(lines) - 1  # Exclude header
            
            print(f"   Result: {row_count} rows, {len(original_result)} characters")
            
            # If > 5 rows, format with CSV
            if row_count > 5:
                print(f"   Large result detected ({row_count} rows), generating CSV...")
                formatted_result = format_result_with_csv(result, max_preview_rows=5)
                print(f"   Formatted result with CSV link")
                print(f"{'='*80}\n")
                return formatted_result
            else:
                # <= 5 rows, return original format (or converted if dict format)
                if is_dict_format:
                    # Return the converted pipe-separated format for consistency
                    result_preview = result[:300] + "..." if len(result) > 300 else result
                    print(f"   Result preview: {result_preview}")
                    print(f"{'='*80}\n")
                    return result
                else:
                    # Return original string format
                    result_preview = original_result[:300] + "..." if len(original_result) > 300 else original_result
                    print(f"   Result preview: {result_preview}")
                    print(f"{'='*80}\n")
                    return original_result
        else:
            # COUNT or other queries - return as-is
            result_preview = result[:300] + "..." if len(result) > 300 else result
            print(f"   Result: {len(result)} characters")
            print(f"   Preview: {result_preview}")
            print(f"{'='*80}\n")
            return result


def _is_restricted_user_query(user_question: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a user's question/request is asking for restricted sensitive data.
    
    This function analyzes the user's natural language question to detect if they're
    asking for direct data from sensitive tables like admin, user_device_assignment, etc.
    
    Args:
        user_question: The user's natural language question
        
    Returns:
        Tuple of (is_restricted: bool, reason: Optional[str])
    """
    question_lower = user_question.lower()
    
    # Patterns that indicate user is asking for direct data from sensitive tables
    restricted_patterns = [
        # Admin table patterns
        (r'\badmin\s+(entry|data|row|record|list|table|information|details)', 'admin'),
        (r'\b(entry|data|row|record|list|table|information|details)\s+.*\badmin\b', 'admin'),
        (r'\b(\d+)(st|nd|rd|th)?\s+(admin|entry|row|record)', 'admin'),
        (r'\b(second|third|fourth|fifth)\s+(admin|entry|row|record)', 'admin'),
        (r'\bgive\s+me\s+admin', 'admin'),
        (r'\bshow\s+me\s+admin', 'admin'),
        (r'\bget\s+admin', 'admin'),
        
        # User/assignment table patterns
        (r'\buser_device_assignment\s+(entry|data|row|record|list|table)', 'user_device_assignment'),
        (r'\b(entry|data|row|record|list|table).*\buser_device_assignment\b', 'user_device_assignment'),
        (r'\buser\s+assignment\s+(entry|data|row|record|list)', 'user_device_assignment'),
        
        # Generic patterns for asking for raw table data
        (r'\bgive\s+me\s+.*\s+(entry|entries|row|rows|record|records|data|table)\s+data', None),
        (r'\bshow\s+me\s+.*\s+(entry|entries|row|rows|record|records|data|table)\s+data', None),
        (r'\bget\s+.*\s+(entry|entries|row|rows|record|records|data|table)\s+data', None),
        (r'\b(\d+)(st|nd|rd|th)?\s+(entry|row|record)\s+data', None),
    ]
    
    for pattern, table in restricted_patterns:
        if re.search(pattern, question_lower):
            return True, table or "sensitive table"
    
    return False, None


def create_check_user_query_restriction_tool():
    """
    Create the check_user_query_restriction tool function.
    This tool validates if the user's question/request is asking for restricted data.
    The LLM should call this tool FIRST with the user's question before proceeding.
    
    Returns:
        Tool function for checking user query restrictions
    """
    @tool
    def check_user_query_restriction(user_question: str) -> str:
        """
        Check if the user's question/request is asking for restricted sensitive data.
        Call this tool FIRST with the user's original question before generating any SQL.
        
        This tool validates if the user is asking for direct data from sensitive system tables
        (like admin, user_device_assignment, etc.). If the question is restricted, it will return
        an error message. If allowed, it will return "User query is allowed. You can proceed."
        
        Args:
            user_question: The user's natural language question/request
        
        Returns:
            "User query is allowed. You can proceed." if allowed,
            or "Sorry, I cannot provide that information." if restricted
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: check_user_query_restriction")
        print(f"{'='*80}")
        print(f"   User Question: {user_question}")
        
        # Validate user question for restricted data requests
        is_restricted, reason = _is_restricted_user_query(user_question)
        if is_restricted:
            error_msg = "Sorry, I cannot provide that information."
            print(f"   ⚠️  BLOCKED: User is asking for restricted data ({reason})")
            print(f"   Response: {error_msg}")
            print(f"{'='*80}\n")
            return error_msg
        
        allowed_msg = "User query is allowed. You can proceed."
        print(f"   ✅ ALLOWED: User query does not request restricted data")
        print(f"   Response: {allowed_msg}")
        print(f"{'='*80}\n")
        return allowed_msg
    
    return check_user_query_restriction


def create_execute_db_query_tool(db: SQLDatabase, vector_store_manager):
    """
    Create the execute_db_query tool function.
    
    Args:
        db: SQLDatabase instance
        vector_store_manager: VectorStoreService instance
        
    Returns:
        Tool function for executing SQL queries
    """
    query_tool = QuerySQLDatabaseTool(db)
    
    @tool
    def execute_db_query(query: str) -> str:
        """
        Execute a SQL query against the PostgreSQL database.
        
        IMPORTANT: You MUST call check_user_query_restriction FIRST with the user's question before calling this tool.
        
        Use this tool AFTER you have:
        1. Called check_user_query_restriction with the user's question and received confirmation
        2. Retrieved examples using get_few_shot_examples (if needed)
        3. Generated a valid PostgreSQL query
        
        Args:
            query: A syntactically correct PostgreSQL query
            
        Returns:
            Query results or error message
        """
        return query_tool.execute(query)
    
    return execute_db_query


def create_get_table_list_tool(db: SQLDatabase, table_metadata: Optional[List[Dict[str, Any]]] = None):
    """
    Create the get_table_list tool function.
    
    Args:
        db: SQLDatabase instance (kept for compatibility, not used)
        table_metadata: Optional list of table metadata dictionaries.
                       If not provided, uses TABLE_METADATA from config.
                       Each dict should have:
                       - name: Table name
                       - description: Table description/use case
                       - important_fields: List of important field names
    
    Returns:
        Tool function for retrieving table list with descriptions and important fields
    """
    # Use provided metadata or fall back to config
    metadata = table_metadata if table_metadata is not None else TABLE_METADATA
    
    @tool
    def get_table_list() -> str:
        """
        FALLBACK TOOL ONLY - Use this ONLY if you have already tried to generate a query from the examples provided in the system prompt and failed.
        
        Do NOT call this tool if you can generate a query from the examples. This is a LAST RESORT tool.
        
        Get a list of all available tables with their descriptions (uses) and important fields.
        
        Use this tool ONLY when:
        - You have already tried to generate a query using the examples in the system prompt
        - You have already tried using get_few_shot_examples to get more examples
        - You STILL cannot generate a query and need to discover what tables are available
        
        This tool returns all configured tables with:
        - Table names
        - Table descriptions/uses (manually configured)
        - Important fields (manually configured)
        
        If you need detailed column information for specific tables, use get_table_structure tool after this.
        
        Returns:
            A formatted string containing table information for all available tables
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: get_table_list")
        print(f"{'='*80}")
        print(f"   Returning all configured tables")
        
        try:
            if not metadata:
                error_msg = "No table metadata configured. Please configure TABLE_METADATA in app/config/table_metadata.py"
                print(f"   ❌ {error_msg}")
                print(f"{'='*80}\n")
                return error_msg
            
            metadata_to_use = metadata
            
            result_parts = []
            result_parts.append("=== AVAILABLE TABLES ===\n")
            
            for table_info in metadata_to_use:
                table_name = table_info.get("name", "")
                description = table_info.get("description", "No description available")
                important_fields = table_info.get("important_fields", [])
                
                if not table_name:
                    continue
                
                result_parts.append(f"\nTable: {table_name}")
                result_parts.append(f"Description: {description}")
                
                if important_fields:
                    fields_str = ', '.join(important_fields) if isinstance(important_fields, list) else str(important_fields)
                    result_parts.append(f"Important Fields: {fields_str}")
                else:
                    result_parts.append("Important Fields: (none specified)")
            
            result = "\n".join(result_parts)
            result_preview = result[:500] + "..." if len(result) > 500 else result
            print(f"   Result: {len(metadata_to_use)} table(s) found")
            print(f"   Preview: {result_preview}")
            print(f"{'='*80}\n")
            
            return result
            
        except Exception as e:
            error_msg = f"Error retrieving table list: {str(e)}"
            print(f"   ❌ {error_msg}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
            return error_msg
    
    return get_table_list


def create_get_table_structure_tool(db: SQLDatabase):
    """
    Create the get_table_structure tool function.
    
    Args:
        db: SQLDatabase instance
        
    Returns:
        Tool function for retrieving table structure
    """
    @tool
    def get_table_structure(table_names: str) -> str:
        """
        FALLBACK TOOL ONLY - Use this ONLY after you have:
        1. Tried to generate a query from examples in the system prompt
        2. Tried using get_few_shot_examples to get more examples
        3. Used get_table_list to discover tables
        4. STILL need detailed column information to construct a query
        
        Do NOT call this tool if you can generate a query from the examples. This is a LAST RESORT tool.
        
        Get the full column structure for specified tables.
        
        Use this tool ONLY when:
        - You have already used get_table_list and identified which tables you need
        - You need detailed column information (names and data types) to construct a query
        - The examples provided don't contain enough schema information
        
        Args:
            table_names: Comma-separated list of table names (e.g., "users,devices,facilities")
                         or a single table name (e.g., "users")
        
        Returns:
            A formatted string containing table name and all columns with their data types
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: get_table_structure")
        print(f"{'='*80}")
        print(f"   Input Table Names: {table_names}")
        
        try:
            # Parse table names (handle comma-separated or single table)
            table_list = [t.strip() for t in table_names.split(',') if t.strip()]
            
            if not table_list:
                error_msg = "No table names provided. Please provide at least one table name."
                print(f"   ❌ {error_msg}")
                print(f"{'='*80}\n")
                return error_msg
            
            # Get database connection
            engine = db._engine
            
            result_parts = []
            result_parts.append("=== TABLE STRUCTURES ===\n")
            
            with engine.connect() as conn:
                for table_name in table_list:
                    # Check if table exists
                    check_query = text("""
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = :table_name
                        )
                    """)
                    exists_result = conn.execute(check_query, {"table_name": table_name})
                    table_exists = exists_result.fetchone()[0]
                    
                    if not table_exists:
                        result_parts.append(f"\nTable: {table_name}")
                        result_parts.append("ERROR: Table does not exist\n")
                        continue
                    
                    # Get all columns with their data types
                    columns_query = text("""
                        SELECT 
                            column_name,
                            data_type,
                            character_maximum_length,
                            is_nullable,
                            column_default
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND table_name = :table_name
                        ORDER BY ordinal_position
                    """)
                    cols_result = conn.execute(columns_query, {"table_name": table_name})
                    columns = cols_result.fetchall()
                    
                    if not columns:
                        result_parts.append(f"\nTable: {table_name}")
                        result_parts.append("No columns found\n")
                        continue
                    
                    result_parts.append(f"\nTable: {table_name}")
                    result_parts.append("Columns:")
                    
                    for col in columns:
                        col_name = col[0]
                        data_type = col[1]
                        max_length = col[2]
                        is_nullable = col[3]
                        default_val = col[4]
                        
                        # Format data type with length if applicable
                        if max_length:
                            type_str = f"{data_type}({max_length})"
                        else:
                            type_str = data_type
                        
                        # Build column description
                        col_desc = f"  - {col_name} ({type_str})"
                        
                        if is_nullable == 'NO':
                            col_desc += " [NOT NULL]"
                        
                        if default_val:
                            # Truncate long defaults
                            default_str = str(default_val)
                            if len(default_str) > 30:
                                default_str = default_str[:30] + "..."
                            col_desc += f" [DEFAULT: {default_str}]"
                        
                        result_parts.append(col_desc)
            
            result = "\n".join(result_parts)
            result_preview = result[:500] + "..." if len(result) > 500 else result
            print(f"   Result: Retrieved structure for {len(table_list)} table(s)")
            print(f"   Preview: {result_preview}")
            print(f"{'='*80}\n")
            
            return result
            
        except Exception as e:
            error_msg = f"Error retrieving table structure: {str(e)}"
            print(f"   ❌ {error_msg}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
            return error_msg
    
    return get_table_structure


def create_journey_list_tool(db: SQLDatabase, user_id: Optional[str] = None):
    """
    Create the journey_list_tool function for journey lists and facility breakdowns.
    
    This tool:
    1. Executes SQL to fetch raw geofencing rows
    2. Runs Python journey calculation algorithm
    3. Returns structured journey list (NOT raw SQL rows)
    
    Args:
        db: SQLDatabase instance
        user_id: User ID for access control
        
    Returns:
        Tool function for calculating journey lists
    """
    query_tool = QuerySQLDatabaseTool(db)
    
    @tool
    def journey_list_tool(sql: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Calculate and return journey list from geofencing data.
        
        IMPORTANT: This tool is for JOURNEY-RELATED questions only.
        Use this tool when the user asks about:
        - Journey lists
        - Facility to facility breakdowns
        - Device-level journeys
        - Movement between facilities
        
        This tool:
        1. Executes the provided SQL query to fetch raw geofencing rows
        2. Runs Python journey calculation algorithm (NOT SQL)
        3. Returns structured journey data with facility details
        
        The SQL query MUST:
        - Use table: device_geofencings (alias: dg) - NOT "geofencing"
        - Join: user_device_assignment (alias: uda) ON uda.device = dg.device_id
        - Filter: WHERE uda.user_id = [user_id]
        - Select: dg.device_id, dg.facility_id, dg.facility_type, dg.entry_event_time, dg.exit_event_time
        - Optional: LEFT JOIN facilities f ON dg.facility_id = f.facility_id (for facility_name)
        - Order: ORDER BY dg.entry_event_time ASC
        
        Example SQL structure:
        SELECT dg.device_id, dg.facility_id, dg.facility_type, f.facility_name, dg.entry_event_time, dg.exit_event_time
        FROM device_geofencings dg
        JOIN user_device_assignment uda ON uda.device = dg.device_id
        LEFT JOIN facilities f ON dg.facility_id = f.facility_id
        WHERE uda.user_id = [user_id] AND dg.device_id = '[device_id]'
        ORDER BY dg.entry_event_time ASC
        
        Args:
            sql: SQL query to fetch geofencing rows (SELECT only)
            params: Optional parameters dict with:
                - start_date: Start date filter (optional)
                - end_date: End date filter (optional)
                - device_id: Specific device ID (optional)
                - extraJourneyTimeLimit: Extra hours for same-facility journeys (optional)
                - offset: Pagination offset (optional)
                - from_facility: Filter journeys that START from this facility ID (optional)
        
        Returns:
            JSON string with:
            - facilities_details: Dict mapping facility_id to facility info
            - journies: List of journey objects with from_facility, to_facility, device_id, times
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: journey_list_tool")
        print(f"{'='*80}")
        print(f"   SQL Query: {sql}")
        print(f"   Params: {params}")
        
        # Validate SQL is SELECT only
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith('SELECT'):
            error_msg = "Only SELECT queries are allowed for journey calculations"
            print(f"   ❌ {error_msg}")
            print(f"{'='*80}\n")
            return error_msg
        
        # Execute query
        try:
            output = query_tool.db.run_no_throw(sql, include_columns=True)
            
            if not output:
                print(f"   Result: No geofencing rows returned")
                print(f"{'='*80}\n")
                return json.dumps({
                    "facilities_details": {},
                    "journies": []
                })
            
            # Debug: Log the raw SQL output format
            print(f"   Raw SQL output type: {type(output)}")
            print(f"   Raw SQL output length: {len(str(output)) if output else 0}")
            print(f"   Raw SQL output preview (first 500 chars): {str(output)[:500] if output else 'None'}")
            
            # Parse query results into list of dicts
            geofencing_rows = _parse_sql_result_to_dicts(output)
            
            print(f"   Parsed {len(geofencing_rows)} geofencing rows from SQL result")
            
            if not geofencing_rows:
                print(f"   ⚠️ WARNING: SQL returned data but parsing failed!")
                print(f"   Raw output was: {str(output)[:1000] if output else 'None'}")
                print(f"{'='*80}\n")
                # Return error with raw data info
                return json.dumps({
                    "error": "Failed to parse SQL results",
                    "raw_output_preview": str(output)[:500] if output else None,
                    "facilities_details": {},
                    "journies": []
                })
            
            # Extract parameters
            extra_journey_time_limit = None
            from_facility = None
            if params:
                extra_journey_time_limit = params.get("extraJourneyTimeLimit")
                from_facility = params.get("from_facility")
            
            # Run Python journey calculation
            filter_note = f" (filtering from_facility={from_facility})" if from_facility else ""
            print(f"   Processing {len(geofencing_rows)} geofencing rows with Python journey algorithm{filter_note}...")
            journey_result = calculate_journey_list(geofencing_rows, extra_journey_time_limit, from_facility)
            
            journey_count = len(journey_result.get('journies', []))
            facilities_count = len(journey_result.get('facilities_details', {}))
            
            # Format result with CSV if > 5 journeys
            if journey_count > 5:
                print(f"   📊 Large result detected ({journey_count} journeys), generating CSV...")
                result_json = format_journey_list_with_csv(journey_result, max_preview=5)
                print(f"   ✅ Showing first 5 journeys, CSV available for all {journey_count} journeys")
            else:
                # Format result normally (no CSV needed)
                result_json = json.dumps(journey_result, indent=2, default=str)
                print(f"   ✅ Showing all {journey_count} journeys (no CSV needed)")
            
            print(f"   Found {facilities_count} unique facilities")
            print(f"{'='*80}\n")
            
            return result_json
            
        except Exception as e:
            error_msg = f"Error calculating journeys: {str(e)}"
            print(f"   ❌ {error_msg}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
            return error_msg
    
    return journey_list_tool


def create_journey_count_tool(db: SQLDatabase, user_id: Optional[str] = None):
    """
    Create the journey_count_tool function for journey counts.
    
    This tool:
    1. Executes SQL to fetch raw geofencing rows
    2. Runs Python journey count algorithm
    3. Returns journey counts by facility pair
    
    Args:
        db: SQLDatabase instance
        user_id: User ID for access control
        
    Returns:
        Tool function for calculating journey counts
    """
    query_tool = QuerySQLDatabaseTool(db)
    
    @tool
    def journey_count_tool(sql: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Calculate and return journey counts from geofencing data.
        
        IMPORTANT: This tool is for JOURNEY COUNT questions only.
        Use this tool when the user asks:
        - "How many journeys..."
        - "Total journeys..."
        - "Count between facilities"
        - "Number of journeys from X to Y"
        
        This tool:
        1. Executes the provided SQL query to fetch raw geofencing rows
        2. Runs Python journey count algorithm (NOT SQL)
        3. Returns journey counts by facility pair
        
        The SQL query MUST:
        - Use table: device_geofencings (alias: dg) - NOT "geofencing"
        - Join: user_device_assignment (alias: uda) ON uda.device = dg.device_id
        - Filter: WHERE uda.user_id = [user_id]
        - Select: dg.device_id, dg.facility_id, dg.facility_type, dg.entry_event_time, dg.exit_event_time
        - Order: ORDER BY dg.entry_event_time ASC
        
        Example SQL structure:
        SELECT dg.device_id, dg.facility_id, dg.facility_type, dg.entry_event_time, dg.exit_event_time
        FROM device_geofencings dg
        JOIN user_device_assignment uda ON uda.device = dg.device_id
        WHERE uda.user_id = [user_id] [AND filters...]
        ORDER BY dg.entry_event_time ASC
        
        Args:
            sql: SQL query to fetch geofencing rows (SELECT only)
            params: Optional parameters dict with:
                - start_date: Start date filter (optional)
                - end_date: End date filter (optional)
                - device_id: Specific device ID (optional)
                - extraJourneyTimeLimit: Extra hours for same-facility journeys (optional)
        
        Returns:
            JSON string with:
            - counts: Dict mapping "facilityA||facilityB" to count
            - total: Total number of journeys
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: journey_count_tool")
        print(f"{'='*80}")
        print(f"   SQL Query: {sql}")
        print(f"   Params: {params}")
        
        # Validate SQL is SELECT only
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith('SELECT'):
            error_msg = "Only SELECT queries are allowed for journey calculations"
            print(f"   ❌ {error_msg}")
            print(f"{'='*80}\n")
            return error_msg
        
        # Execute query
        try:
            output = query_tool.db.run_no_throw(sql, include_columns=True)
            
            if not output:
                print(f"   Result: No geofencing rows returned")
                print(f"{'='*80}\n")
                return json.dumps({
                    "counts": {},
                    "total": 0
                })
            
            # Debug: Log the raw SQL output format
            print(f"   Raw SQL output type: {type(output)}")
            print(f"   Raw SQL output length: {len(str(output)) if output else 0}")
            print(f"   Raw SQL output preview (first 500 chars): {str(output)[:500] if output else 'None'}")
            
            # Parse query results into list of dicts
            geofencing_rows = _parse_sql_result_to_dicts(output)
            
            print(f"   Parsed {len(geofencing_rows)} geofencing rows from SQL result")
            
            if not geofencing_rows:
                print(f"   ⚠️ WARNING: SQL returned data but parsing failed!")
                print(f"   Raw output was: {str(output)[:1000] if output else 'None'}")
                print(f"{'='*80}\n")
                # Return error with raw data info
                return json.dumps({
                    "error": "Failed to parse SQL results",
                    "raw_output_preview": str(output)[:500] if output else None,
                    "counts": {},
                    "total": 0
                })
            
            # Extract extraJourneyTimeLimit from params
            extra_journey_time_limit = None
            if params:
                extra_journey_time_limit = params.get("extraJourneyTimeLimit")
            
            # Run Python journey calculation
            print(f"   Processing {len(geofencing_rows)} geofencing rows with Python journey algorithm...")
            journey_result = calculate_journey_counts(geofencing_rows, extra_journey_time_limit)
            
            # Log metadata for debugging
            metadata = journey_result.get('metadata', {})
            if metadata:
                print(f"   📊 Metadata: {metadata.get('total_rows_processed', 0)} rows, "
                      f"{metadata.get('devices_processed', 0)} devices, "
                      f"facility types: {metadata.get('facility_types_found', [])}")
            
            # Format result
            result_json = json.dumps(journey_result, indent=2, default=str)
            
            total_journeys = journey_result.get('total', 0)
            print(f"   ✅ Calculated {total_journeys} total journeys")
            print(f"   Found {len(journey_result.get('counts', {}))} unique facility pairs")
            
            if total_journeys == 0 and len(geofencing_rows) > 0:
                print(f"   ⚠️ NOTE: Found {len(geofencing_rows)} geofencing records but 0 journeys")
                print(f"   This could mean: same facility only, or journey time < 4 hours")
            
            print(f"{'='*80}\n")
            
            return result_json
            
        except Exception as e:
            error_msg = f"Error calculating journey counts: {str(e)}"
            print(f"   ❌ {error_msg}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
            return error_msg
    
    return journey_count_tool


def _parse_sql_result_to_dicts(sql_result: str) -> List[Dict[str, Any]]:
    """
    Parse SQL result string into list of dictionaries.
    
    Handles different SQL result formats from SQLDatabase.run_no_throw().
    
    Args:
        sql_result: SQL result string (can be pipe-separated, JSON, or Python list format)
        
    Returns:
        List of dictionaries with row data
    """
    rows = []
    
    if not sql_result or not sql_result.strip():
        return rows
    
    # Try to parse as Python list/dict format first
    import ast
    import datetime as dt_module
    from datetime import datetime
    try:
        if sql_result.strip().startswith('['):
            # First try ast.literal_eval (safe, but doesn't handle datetime objects)
            try:
                parsed = ast.literal_eval(sql_result)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                # If that fails, it might contain datetime objects
                # Use eval() with a safe namespace (only datetime module and class allowed)
                # This handles cases like: [{'entry_event_time': datetime.datetime(2024, 12, 6, 15, 14, 59), ...}]
                # The string uses datetime.datetime(...) so we need both the module and the class
                safe_dict = {
                    '__builtins__': {},
                    'datetime': dt_module,  # Provide the datetime module so datetime.datetime works
                    'dict': dict,
                    'list': list,
                    'tuple': tuple,
                    'None': None,
                    'True': True,
                    'False': False
                }
                parsed = eval(sql_result, safe_dict)
                if isinstance(parsed, list):
                    # datetime objects are preserved - perfect!
                    return parsed
    except (ValueError, SyntaxError, NameError, TypeError) as e:
        logger.debug(f"Failed to parse as Python list: {e}")
        pass
    
    # Try to parse as JSON
    try:
        parsed = json.loads(sql_result)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Parse as pipe-separated format (default SQLDatabase format)
    lines = sql_result.strip().split('\n')
    if len(lines) < 2:
        return rows
    
    # First line is headers
    headers = [h.strip() for h in lines[0].split('|')]
    headers = [h for h in headers if h]  # Remove empty headers
    
    # Remaining lines are data
    for line in lines[1:]:
        if not line.strip():
            continue
        
        values = [v.strip() for v in line.split('|')]
        values = [v for v in values if v]  # Remove empty values
        
        if len(values) != len(headers):
            continue
        
        row_dict = {}
        for i, header in enumerate(headers):
            value = values[i] if i < len(values) else None
            
            # Try to convert numeric values or parse timestamps
            if value:
                # Check if it looks like a timestamp string (for entry_event_time, exit_event_time columns)
                if isinstance(value, str) and ('event_time' in header.lower() or 'time' in header.lower()):
                    # Try to parse as timestamp (common PostgreSQL formats)
                    from datetime import datetime
                    timestamp_formats = [
                        '%Y-%m-%d %H:%M:%S.%f',
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S.%f',
                        '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%d %H:%M:%S+00:00',
                        '%Y-%m-%d %H:%M:%S.%f+00:00'
                    ]
                    timestamp_parsed = False
                    for fmt in timestamp_formats:
                        try:
                            value = datetime.strptime(value, fmt)
                            timestamp_parsed = True
                            break
                        except ValueError:
                            continue
                    
                    # If timestamp parsing failed, keep as string (will be handled by _convert_to_unix_timestamp)
                    if not timestamp_parsed:
                        # Keep as string - _convert_to_unix_timestamp will handle it
                        pass
                elif isinstance(value, str):
                    # For non-timestamp strings, try numeric conversion
                    try:
                        # Try float first (handles decimals)
                        if '.' in value and not any(c.isalpha() for c in value):
                            value = float(value)
                        elif not any(c.isalpha() for c in value):
                            # Try int
                            value = int(value)
                    except ValueError:
                        # Keep as string
                        pass
                # If it's already a datetime, int, or float, keep as is
            
            row_dict[header] = value
        
        rows.append(row_dict)
    
    return rows

