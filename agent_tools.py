"""
LangGraph Agent Tools

This module defines tools that the agent can use:
1. get_few_shot_examples - Retrieves similar examples from FAISS
2. execute_db_query - Executes SQL queries against PostgreSQL
"""

from typing import Optional, Dict, Any, Sequence
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy.engine import Result


def create_get_few_shot_examples_tool(vector_store_manager):
    """
    Create the get_few_shot_examples tool function.
    
    Args:
        vector_store_manager: VectorStoreManager instance
        
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


def create_execute_db_query_tool(db: SQLDatabase, vector_store_manager):
    """
    Create the execute_db_query tool function.
    
    Args:
        db: SQLDatabase instance
        vector_store_manager: VectorStoreManager instance
        
    Returns:
        Tool function for executing SQL queries
    """
    query_tool = QuerySQLDatabaseTool(db)
    
    @tool
    def execute_db_query(query: str) -> str:
        """
        Execute a SQL query against the PostgreSQL database.
        
        Use this tool AFTER you have:
        1. Retrieved examples using get_few_shot_examples (if needed)
        2. Generated a valid PostgreSQL query
        3. Validated the query structure
        
        Args:
            query: A syntactically correct PostgreSQL query
            
        Returns:
            Query results or error message
        """
        return query_tool.execute(query)
    
    return execute_db_query

