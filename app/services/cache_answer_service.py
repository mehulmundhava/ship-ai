"""
Cache Answer Service

Handles vector-based caching of LLM answers with strict parameter matching.
Ensures cached answers are only returned when SQL parameters match exactly.
"""

import json
import re
from typing import Dict, List, Optional, Any
from sqlalchemy import text
from langchain_community.utilities.sql_database import SQLDatabase
from app.config.database import sync_engine
from app.config.settings import settings
from app.services.vector_store_service import VectorStoreService
from app.core.agent.agent_tools import QuerySQLDatabaseTool, create_journey_list_tool, create_journey_count_tool
import logging

logger = logging.getLogger("ship_rag_ai")


def extract_params_from_question(question: str) -> Dict[str, List[str]]:
    """
    Extract parameters from question text as a fallback when SQL is not available.
    This is used ONLY when sql_query is None (cache check before LLM runs).
    
    Args:
        question: User's question text
        
    Returns:
        Dictionary with parameter names as keys and lists of values as values.
        Returns empty dict {} if no parameters found.
        
    Extracts:
        - device_id (from patterns like "device WT01...", "device_id WT01...")
        - facility_id (from patterns like "facility F123...")
        - from_facility (from patterns like "from facility F123...")
        - to_facility (from patterns like "to facility F123...")
    """
    if not question:
        return {}
    
    params = {}
    question_upper = question.upper()
    
    # Extract device_id from question text
    # Patterns: "device WT01...", "device_id WT01...", "for device WT01...", etc.
    # Device IDs are typically WT01 followed by 10-15 alphanumeric characters
    device_patterns = [
        r'(?:for\s+)?device(?:\s+id)?\s+(WT01[A-Z0-9]{10,})',  # "device WT01...", "for device WT01...", "device id WT01..."
        r'device_id\s+(WT01[A-Z0-9]{10,})',  # "device_id WT01..."
        r'(?:asset|asset_id)\s+(WT01[A-Z0-9]{10,})',  # "asset WT01..."
    ]
    
    device_values = []
    for pattern in device_patterns:
        matches = re.finditer(pattern, question, re.IGNORECASE)
        for match in matches:
            device_id = match.group(1)
            if device_id and device_id not in device_values:
                device_values.append(device_id)
    
    if device_values:
        params['device_id'] = sorted(list(set(device_values)))
    
    # Extract facility_id from question text
    # Patterns: "facility F123...", "facility_id F123..."
    facility_patterns = [
        r'facility\s+([A-Z]{1,5}[A-Z0-9]{5,})',  # "facility DOPTZ00001"
        r'facility_id\s+([A-Z]{1,5}[A-Z0-9]{5,})',  # "facility_id DOPTZ00001"
    ]
    
    facility_values = []
    for pattern in facility_patterns:
        matches = re.finditer(pattern, question_upper, re.IGNORECASE)
        for match in matches:
            facility_id = match.group(1)
            if facility_id and facility_id not in facility_values:
                facility_values.append(facility_id)
    
    if facility_values:
        params['facility_id'] = sorted(list(set(facility_values)))
    
    # Extract from_facility and to_facility (for journey questions)
    # Patterns: "from facility F123", "starting from facility F123", "from F123"
    from_facility_pattern = r'(?:starting\s+)?from\s+(?:facility\s+)?([A-Z]{1,5}[A-Z0-9]{5,})'
    from_matches = re.finditer(from_facility_pattern, question_upper, re.IGNORECASE)
    from_values = []
    for match in from_matches:
        facility_id = match.group(1)
        if facility_id and facility_id not in from_values:
            from_values.append(facility_id)
    
    if from_values:
        params['from_facility'] = sorted(list(set(from_values)))
    
    to_facility_pattern = r'to\s+(?:facility\s+)?([A-Z]{1,5}[A-Z0-9]{5,})'
    to_matches = re.finditer(to_facility_pattern, question_upper, re.IGNORECASE)
    to_values = []
    for match in to_matches:
        facility_id = match.group(1)
        if facility_id and facility_id not in to_values:
            to_values.append(facility_id)
    
    if to_values:
        params['to_facility'] = sorted(list(set(to_values)))
    
    return params


def extract_sql_parameters(sql: str) -> Dict[str, List[str]]:
    """
    Extract parameters from SQL query.
    Uses SQL as the ONLY source of truth for parameters.
    Supports table aliases (e.g., dg.device_id, cd.facility_id).
    
    Args:
        sql: SQL query string
        
    Returns:
        Dictionary with parameter names as keys and lists of values as values.
        Returns empty dict {} if no parameters found.
        
    Extracts:
        - device_id
        - facility_id
        - from_facility
        - to_facility
        - interval (INTERVAL '180 days', etc)
    """
    if not sql:
        return {}
    
    params = {}
    
    # Pattern to match parameter with optional table alias
    # Matches: device_id, dg.device_id, cd.facility_id, etc.
    alias_pattern = r'(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?'
    
    def extract_param_values(param_name: str, sql_text: str) -> List[str]:
        """Extract all values for a parameter from SQL."""
        values = []
        
        # Pattern for simple equality: device_id = 'value' or device_id='value'
        # Exclude JOIN conditions (column references like f.facility_id, dg.device_id)
        simple_pattern = rf'{alias_pattern}{param_name}\s*=\s*([\'"]?)([^\'"\s,\)]+)\1'
        for match in re.finditer(simple_pattern, sql_text, re.IGNORECASE):
            value = match.group(2).strip("'\"")
            # Filter out column references (values containing a dot, like f.facility_id, dg.device_id)
            # These are JOIN conditions, not parameter values
            if value and '.' in value and not value.startswith("'") and not value.startswith('"'):
                continue
            if value and value not in values:
                values.append(value)
        
        # Pattern for IN clause: device_id IN ('val1', 'val2') or device_id IN (val1, val2)
        in_pattern = rf'{alias_pattern}{param_name}\s+IN\s*\(([^)]+)\)'
        for match in re.finditer(in_pattern, sql_text, re.IGNORECASE):
            in_values_str = match.group(1)
            # Extract individual values from IN clause
            in_values = re.findall(r"['\"]?([^,'\"]+)['\"]?", in_values_str)
            for val in in_values:
                val = val.strip("'\"")
                if val and val not in values:
                    values.append(val)
        
        # Pattern for comparison operators: device_id > 'value', device_id < 'value', etc.
        comparison_pattern = rf'{alias_pattern}{param_name}\s*(?:!=|<>|>|<|>=|<=|LIKE|ILIKE)\s*([\'"]?)([^\'"\s,\)]+)\1'
        for match in re.finditer(comparison_pattern, sql_text, re.IGNORECASE):
            value = match.group(2).strip("'\"")
            if value and value not in values:
                values.append(value)
        
        return sorted(list(set(values)))
    
    # Extract device_id
    device_values = extract_param_values('device_id', sql)
    if device_values:
        params['device_id'] = device_values
    
    # Extract facility_id
    facility_values = extract_param_values('facility_id', sql)
    if facility_values:
        params['facility_id'] = facility_values
    
    # Extract from_facility
    from_facility_values = extract_param_values('from_facility', sql)
    if from_facility_values:
        params['from_facility'] = from_facility_values
    
    # Extract to_facility
    to_facility_values = extract_param_values('to_facility', sql)
    if to_facility_values:
        params['to_facility'] = to_facility_values
    
    # Extract interval (INTERVAL '180 days', INTERVAL '30 days', etc)
    interval_pattern = r"INTERVAL\s+['\"]([^'\"]+)['\"]"
    interval_matches = re.finditer(interval_pattern, sql, re.IGNORECASE)
    interval_values = []
    for match in interval_matches:
        value = match.group(1).strip()
        if value and value not in interval_values:
            interval_values.append(value)
    if interval_values:
        params['interval'] = sorted(list(set(interval_values)))
    
    return params


def parameters_match(p1: Dict[str, List[str]], p2: Dict[str, List[str]]) -> bool:
    """
    Check if two parameter dictionaries match exactly.
    
    Rules:
        - Both empty {} → True (generic question)
        - Both non-empty AND exactly equal → True
        - One empty, one non-empty → False
        - Both non-empty but differ → False
        
    Args:
        p1: First parameter dictionary
        p2: Second parameter dictionary
        
    Returns:
        True if parameters match, False otherwise
    """
    # Both empty → OK
    if not p1 and not p2:
        return True
    
    # One empty, one non-empty → REJECT
    if not p1 or not p2:
        return False
    
    # Keys must match exactly
    if set(p1.keys()) != set(p2.keys()):
        return False
    
    # Values compared as sets (order doesn't matter)
    for key in p1.keys():
        if set(p1[key]) != set(p2[key]):
            return False
    
    return True


class CacheAnswerService:
    """
    Service for managing cached answers with strict parameter matching.
    """
    
    def __init__(self, vector_store: VectorStoreService, sql_db: Optional[SQLDatabase] = None, user_id: Optional[str] = None):
        """
        Initialize cache service.
        
        Args:
            vector_store: VectorStoreService instance for embeddings
            sql_db: Optional SQLDatabase instance for executing adapted SQL
            user_id: Optional user_id for journey tools
        """
        self.vector_store = vector_store
        self.engine = sync_engine
        self.embedding_field_name = vector_store.embedding_field_name
        self.similarity_threshold = settings.VECTOR_CACHE_SIMILARITY_THRESHOLD
        self.sql_db = sql_db
        self.user_id = user_id

    def check_80_match_and_execute(
        self,
        question: str,
        question_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        80% match and execute: uses only base columns of ai_vector_examples,
        no cache columns. When the user's question is >= threshold similar to an
        example, run that example's SQL (or an adapted version) and return the result
        without calling the LLM. Never reads or writes cache columns.

        Args:
            question: User's question
            question_type: 'journey' or 'non_journey' (for execution path)

        Returns:
            Dict with answer, sql_query, result_data, similarity, original_question
            if a candidate matched and execution succeeded; None otherwise.
        """
        if not self.sql_db:
            logger.debug("80% path: SQL database not available")
            return None
        # 0.80 bar is fixed in code; no env flag. If similarity ≥ 0.80, use example SQL.
        threshold = 0.80
        try:
            query_embedding = self.vector_store.embed_query(question)
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Base-columns-only query: no entry_type, is_active, question_type, user_type
            search_query = text(f"""
                SELECT
                    id,
                    question,
                    sql_query,
                    metadata,
                    {self.embedding_field_name} <-> '{embedding_str}'::vector AS distance,
                    1 - ({self.embedding_field_name} <-> '{embedding_str}'::vector) AS similarity
                FROM ai_vector_examples
                WHERE {self.embedding_field_name} IS NOT NULL
                ORDER BY {self.embedding_field_name} <-> '{embedding_str}'::vector
                LIMIT 3
            """)

            with self.engine.connect() as conn:
                result = conn.execute(search_query)
                candidates = result.fetchall()

            if not candidates:
                logger.debug("80% path: No candidates found in ai_vector_examples")
                return None

            current_params = extract_params_from_question(question)
            top_similarities = [float(row.similarity) if row.similarity else 0.0 for row in candidates[:3]]

            for row in candidates:
                sim = float(row.similarity) if row.similarity else 0.0
                if sim < threshold:
                    logger.debug(f"80% path: Skipping candidate (similarity {sim:.4f} < {threshold})")
                    continue
                sql_query = getattr(row, "sql_query", None)
                if not sql_query:
                    continue

                example_params: Dict[str, List[str]] = {}
                meta = getattr(row, "metadata", None)
                if meta:
                    try:
                        d = meta if isinstance(meta, dict) else json.loads(meta) if isinstance(meta, str) else {}
                        if isinstance(d.get("params"), dict):
                            example_params = {k: v if isinstance(v, list) else [v] for k, v in d["params"].items()}
                    except Exception:
                        pass
                if not example_params:
                    example_params = extract_sql_parameters(sql_query)

                adapted_result = self._try_adapt_and_execute_sql(
                    cached_sql=sql_query,
                    cached_params=example_params,
                    current_params=current_params,
                    question=question,
                    question_type=question_type,
                    similarity=sim,
                )
                if adapted_result:
                    logger.info(f"[80% path] hit similarity={sim:.4f} question_type={question_type}")
                    return adapted_result

            top = top_similarities[0] if top_similarities else 0.0
            logger.info(f"[80% path] miss top={top:.4f} threshold={threshold}")
            return None
        except Exception as e:
            logger.warning(f"80% match-and-execute error: {e}")
            return None

    
    def _adapt_sql_parameters(self, sql: str, old_params: Dict[str, List[str]], new_params: Dict[str, List[str]]) -> Optional[str]:
        """
        Adapt SQL query by replacing old parameters with new parameters.
        
        Args:
            sql: Original SQL query
            old_params: Parameters in the original SQL
            new_params: New parameters to use
            
        Returns:
            Adapted SQL query, or None if adaptation fails
        """
        if not sql:
            return None
        
        try:
            adapted_sql = sql
            
            # Note: from_facility is handled separately - it should be passed as a parameter to journey tools,
            # not added to the SQL WHERE clause. The journey tool will handle filtering journeys that start from that facility.
            
            # Replace each parameter type (excluding from_facility which was handled above)
            for param_name in old_params.keys():
                if param_name == 'from_facility':
                    continue  # Already handled above
                    
                if param_name not in new_params:
                    logger.warning(f"Parameter {param_name} not found in new_params, skipping")
                    continue
                
                old_values = old_params[param_name]
                new_values = new_params[param_name]
                
                if not old_values or not new_values:
                    continue
                
                # For each old value, replace with corresponding new value
                # If multiple values, use the first one (or handle IN clause)
                old_value = old_values[0] if old_values else None
                new_value = new_values[0] if new_values else None
                
                if not old_value or not new_value:
                    continue
                
                # Skip column references (values with dots that aren't quoted)
                if '.' in old_value and not old_value.startswith("'") and not old_value.startswith('"'):
                    logger.debug(f"Skipping column reference: {param_name} = {old_value}")
                    continue
                
                # Pattern to match parameter with optional table alias
                alias_pattern = r'(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?'
                
                # Replace in WHERE clause (equality): device_id = 'old_value'
                # Only replace in WHERE clauses, not in JOIN clauses
                # Look for WHERE before the match to ensure we're in a WHERE clause
                pattern = rf'({alias_pattern}{param_name}\s*=\s*[\'"]?){re.escape(old_value)}([\'"]?\s*)'
                replacement = rf'\1{new_value}\2'
                
                # Only replace if it's in a WHERE clause (not in JOIN)
                def replace_if_in_where(match):
                    full_match = match.group(0)
                    # Check if this is in a WHERE clause by looking backwards
                    before_match = adapted_sql[:match.start()]
                    # Find the last WHERE or JOIN before this match
                    last_where = before_match.rfind('WHERE')
                    last_join = before_match.rfind('JOIN')
                    # Only replace if WHERE comes after the last JOIN (or no JOIN found)
                    if last_where > last_join or last_join == -1:
                        return replacement.replace(r'\1', match.group(1)).replace(r'\2', match.group(2))
                    return full_match
                
                # Use a more careful replacement that checks context
                where_section_pattern = rf'(WHERE\s+[^O]*?)({alias_pattern}{param_name}\s*=\s*[\'"]?){re.escape(old_value)}([\'"]?\s*)'
                def replace_in_where_section(match):
                    prefix = match.group(1)
                    param_part = match.group(2)
                    quote_part = match.group(3)
                    return f"{prefix}{param_part}{new_value}{quote_part}"
                
                adapted_sql = re.sub(where_section_pattern, replace_in_where_section, adapted_sql, flags=re.IGNORECASE | re.DOTALL)
                
                # Replace in WHERE clause (IN clause): device_id IN ('old_value', ...)
                # Match the IN clause and replace the specific value
                in_pattern = rf'({alias_pattern}{param_name}\s+IN\s*\([^)]*?)([\'"]?){re.escape(old_value)}([\'"]?)([^)]*?\))'
                def replace_in_value(match):
                    prefix = match.group(1)
                    quote1 = match.group(2)
                    quote2 = match.group(3)
                    suffix = match.group(4)
                    # Replace old_value with new_value, preserving quotes
                    return f"{prefix}{quote1}{new_value}{quote2}{suffix}"
                
                adapted_sql = re.sub(in_pattern, replace_in_value, adapted_sql, flags=re.IGNORECASE)
            
            logger.info(f"✅ SQL adapted successfully")
            logger.debug(f"Original SQL: {sql[:200]}...")
            logger.debug(f"Adapted SQL: {adapted_sql[:200]}...")
            
            return adapted_sql
            
        except Exception as e:
            logger.error(f"Error adapting SQL: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _try_adapt_and_execute_sql(
        self,
        cached_sql: str,
        cached_params: Dict[str, List[str]],
        current_params: Dict[str, List[str]],
        question: str,
        question_type: str,
        similarity: float
    ) -> Optional[Dict[str, Any]]:
        """
        Try to adapt cached SQL with current parameters and execute it.
        If successful, format answer without full LLM agent.
        
        Args:
            cached_sql: SQL from cached entry
            cached_params: Parameters from cached entry
            current_params: Current request parameters
            question: User's question
            question_type: 'journey' or 'non_journey'
            similarity: Similarity score
            
        Returns:
            Dictionary with answer, SQL query, and results if successful, None otherwise
        """
        if not self.sql_db:
            logger.warning("SQL database not available for SQL adaptation")
            return None
        
        try:
            # Adapt SQL by replacing parameters
            adapted_sql = self._adapt_sql_parameters(cached_sql, cached_params, current_params)
            if not adapted_sql:
                logger.warning("Failed to adapt SQL parameters")
                return None
            
            logger.info(f"🔄 Executing adapted SQL (similarity: {similarity:.4f})")
            
            # Execute SQL based on question type
            query_result = None
            result_data = None
            
            if question_type == 'journey':
                # For journey questions, use journey_list_tool
                try:
                    journey_tool = create_journey_list_tool(self.sql_db, self.user_id)
                    # Prepare params dict - include from_facility if present
                    tool_params = {}
                    if 'from_facility' in current_params and current_params['from_facility']:
                        tool_params['from_facility'] = current_params['from_facility'][0]
                    # LangChain tool - use .invoke() with dictionary input
                    tool_result = journey_tool.invoke({"sql": adapted_sql, "params": tool_params})
                    
                    # Parse JSON result
                    if isinstance(tool_result, str):
                        try:
                            result_data = json.loads(tool_result)
                            query_result = tool_result
                        except json.JSONDecodeError:
                            query_result = tool_result
                            result_data = {"raw_result": tool_result}
                    else:
                        result_data = tool_result if isinstance(tool_result, dict) else {"raw_result": str(tool_result)}
                        query_result = json.dumps(result_data, indent=2, default=str)
                    
                    logger.info(f"✅ Journey tool executed successfully")
                except Exception as e:
                    logger.error(f"Error executing journey tool: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            else:
                # For non-journey questions, use regular SQL execution
                try:
                    query_tool = QuerySQLDatabaseTool(self.sql_db)
                    query_result = query_tool.execute(adapted_sql)
                    
                    # Try to parse result as JSON or dict
                    if isinstance(query_result, str):
                        # Check if it's JSON
                        if query_result.strip().startswith('{') or query_result.strip().startswith('['):
                            try:
                                result_data = json.loads(query_result)
                            except json.JSONDecodeError:
                                result_data = {"raw_result": query_result}
                        else:
                            result_data = {"raw_result": query_result}
                    else:
                        result_data = query_result if isinstance(query_result, dict) else {"raw_result": str(query_result)}
                    
                    logger.info(f"✅ SQL executed successfully")
                except Exception as e:
                    logger.error(f"Error executing SQL: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            # Format answer from results (simple template-based, no LLM)
            answer = self._format_answer_from_results(question, adapted_sql, query_result, result_data, question_type)
            
            if not answer:
                logger.warning("Failed to format answer from results")
                return None
            
            return {
                "answer": answer,
                "sql_query": adapted_sql,
                "result_data": result_data,
                "similarity": similarity,
                "original_question": question,
                "cache_id": None,  # Not from cache, but adapted
                "adapted": True  # Flag to indicate this was adapted
            }
            
        except Exception as e:
            logger.error(f"Error in SQL adaptation and execution: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _format_answer_from_results(
        self,
        question: str,
        sql_query: str,
        query_result: str,
        result_data: Optional[Dict[str, Any]],
        question_type: str
    ) -> Optional[str]:
        """
        Format answer from query results using simple templates.
        This avoids LLM calls for formatting.
        
        Args:
            question: User's question
            sql_query: SQL query that was executed
            query_result: Raw query result string
            result_data: Parsed result data (if available)
            question_type: 'journey' or 'non_journey'
            
        Returns:
            Formatted answer string, or None if formatting fails
        """
        try:
            if question_type == 'journey' and result_data:
                # Handle journey results
                journies = result_data.get('journies', [])
                total_journeys = result_data.get('total_journeys', len(journies))
                csv_download_link = result_data.get('csv_download_link')
                csv_id = result_data.get('csv_id')
                
                if csv_download_link:
                    path = csv_download_link if csv_download_link.startswith("/") else (f"/download-csv/{csv_id}" if csv_id else csv_download_link)
                    return f"The device has a total of {total_journeys} facility-to-facility movements. [Download CSV]({path})"
                elif total_journeys > 0:
                    return f"Found {total_journeys} facility-to-facility movement(s) for the device."
                else:
                    return "No facility-to-facility movements found for the device."
            
            elif result_data:
                # Handle regular SQL results
                # Try to extract meaningful information
                if 'raw_result' in result_data:
                    raw = result_data['raw_result']
                    # Check if it mentions CSV
                    if 'CSV Download Link' in raw:
                        # Extract CSV link and row count (use relative path for UI/Postman)
                        csv_match = re.search(r'CSV Download Link:\s*(/download-csv/[^\s\n]+)', raw)
                        row_match = re.search(r'Total rows:\s*(\d+)', raw)
                        if csv_match and row_match:
                            csv_path = csv_match.group(1)
                            row_count = row_match.group(1)
                            return f"Found {row_count} result(s). [Download CSV]({csv_path})"
                    
                    # Check if it's an empty result
                    if '0 rows' in raw or 'no rows' in raw.lower():
                        return "No results found matching your criteria."
                    
                    # Single-value result (e.g. "max_temperature\n17.75" or "count\n42")
                    q_lower = (question or "").lower()
                    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                    if len(lines) >= 2:
                        # Header line + value line(s); treat first data line as primary value
                        header = lines[0]
                        first_val = lines[1] if len(lines) > 1 else ""
                        if "\t" in first_val:
                            first_val = first_val.split("\t")[0].strip()
                        if first_val.lower() in ("none", "null", ""):
                            if "maximum temperature" in q_lower or "max temperature" in q_lower:
                                return "The maximum temperature for the specified device and date is not available."
                            if "minimum temperature" in q_lower or "min temperature" in q_lower:
                                return "The minimum temperature for the specified device and date is not available."
                            return "No results found matching your criteria."
                        # One numeric/value result with a sensible sentence
                        if "maximum temperature" in q_lower or "max temperature" in q_lower:
                            return f"The maximum temperature was {first_val}°C."
                        if "minimum temperature" in q_lower or "min temperature" in q_lower:
                            return f"The minimum temperature was {first_val}°C."
                        if "count" in header.lower() or "total" in header.lower():
                            return f"The result is {first_val}."
                        return f"The result is {first_val}."
                    
                    # Otherwise, return a generic message
                    return f"Query executed successfully. {raw[:200]}..."
                else:
                    # Structured data
                    return f"Query executed successfully. Found results matching your criteria."
            else:
                # Fallback
                if '0 rows' in query_result or 'no rows' in query_result.lower():
                    return "No results found matching your criteria."
                return f"Query executed successfully. {query_result[:200]}..."
                
        except Exception as e:
            logger.error(f"Error formatting answer: {e}")
            return None
    
