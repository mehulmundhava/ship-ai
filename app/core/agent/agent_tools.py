"""
LangGraph Agent Tools

This module defines tools that the agent can use:
1. get_few_shot_examples - Retrieves similar examples from PostgreSQL vector store (pgvector)
2. execute_db_query - Executes SQL queries against PostgreSQL
"""

from typing import Optional, Dict, Any, Sequence, Tuple
import re
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy.engine import Result

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


class QuerySQLDatabaseTool:
    """
    Tool for executing SQL queries against PostgreSQL database.
    """
    
    def __init__(self, db: SQLDatabase):
        self.db = db
    
    def execute(self, query: str) -> str:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results as string, or error message if query fails
        """
        print(f"\n{'='*80}")
        print(f"🔧 TOOL CALLED: execute_db_query")
        print(f"{'='*80}")
        print(f"   SQL Query: {query}")
        
        # Execute query without throwing exceptions
        output = self.db.run_no_throw(query, include_columns=True)
        
        # Check if query returned no rows
        if not output:
            print(f"   Result: No rows returned")
            print(f"{'='*80}\n")
            return ":::::: Query execution has returned 0 rows. Return final answer accordingly. ::::::"
        
        result = str(output)
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

