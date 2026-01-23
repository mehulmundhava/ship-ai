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
    
    def check_cached_answer(
        self,
        question: str,
        user_type: str,
        question_type: str,
        sql_query: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check for cached answer with strict parameter matching.
        
        STRICT RULE: A cached answer is valid ONLY if:
            - similarity >= threshold
            - AND SQL parameters match EXACTLY
        
        Behavior:
            1. Fetch TOP 3 candidates by similarity
            2. For each candidate:
               - Extract parameters from cached entry SQL
               - Extract parameters from current SQL (if available)
               - Apply safety rules
               - Return first valid match
            3. If none valid → return None (LLM runs)
        
        Args:
            question: User's question
            user_type: 'admin' or 'user'
            question_type: 'journey' or 'non_journey'
            sql_query: Optional SQL query from current request (if available)
            
        Returns:
            Dictionary with cached answer data if match found, None otherwise
        """
        try:
            # Generate embedding for the question
            query_embedding = self.vector_store.embed_query(question)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Extract parameters from current SQL (if available) OR from question text (fallback)
            current_params = {}
            if sql_query:
                # Priority 1: Extract from SQL (most reliable)
                try:
                    current_params = extract_sql_parameters(sql_query)
                    logger.info(f"Extracted current SQL parameters: {current_params}")
                except Exception as e:
                    logger.warning(f"Failed to extract parameters from current SQL: {e}")
                    # If extraction fails on SQL with WHERE clauses, reject cache
                    # But we can't know if it has WHERE clauses without parsing
                    # So we'll be conservative and reject if extraction fails
                    current_params = None  # Signal extraction failure
            else:
                # Priority 2: Extract from question text as fallback (when SQL not available)
                try:
                    current_params = extract_params_from_question(question)
                    if current_params:
                        logger.info(f"Extracted parameters from question text (fallback): {current_params}")
                    else:
                        logger.info("No parameters found in question text - will match non-parameterized entries only")
                except Exception as e:
                    logger.warning(f"Failed to extract parameters from question text: {e}")
                    current_params = {}  # Empty params - will match non-parameterized entries
            
            # Fetch TOP 3 candidates by similarity
            # Include both 'cache' entries AND 'example' entries
            # (training examples can serve as cache candidates even without answer field)
            search_query = text(f"""
                SELECT 
                    id,
                    question,
                    sql_query,
                    answer,
                    metadata,
                    entry_type,
                    question_type as cached_question_type,
                    user_type as cached_user_type,
                    {self.embedding_field_name} <-> '{embedding_str}'::vector AS distance,
                    1 - ({self.embedding_field_name} <-> '{embedding_str}'::vector) AS similarity
                FROM ai_vector_examples
                WHERE is_active = true
                  AND {self.embedding_field_name} IS NOT NULL
                  AND (
                      -- Cache entries: must match question_type and user_type exactly
                      (entry_type = 'cache' 
                       AND question_type = :question_type 
                       AND user_type = :user_type)
                      OR
                      -- Example entries: can be used if they match question_type/user_type (or if NULL, accept them)
                      (entry_type = 'example' 
                       AND (question_type = :question_type OR question_type IS NULL)
                       AND (user_type = :user_type OR user_type IS NULL))
                  )
                ORDER BY {self.embedding_field_name} <-> '{embedding_str}'::vector
                LIMIT 3
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(search_query, {
                    "question_type": question_type,
                    "user_type": user_type
                })
                candidates = result.fetchall()
            
            if not candidates:
                logger.debug("No cache candidates found")
                return None
            
            logger.info(f"Found {len(candidates)} cache candidates")
            
            # Check each candidate
            for candidate in candidates:
                similarity = float(candidate.similarity) if candidate.similarity else 0.0
                entry_type = getattr(candidate, 'entry_type', 'unknown')
                
                logger.info(f"Candidate {candidate.id} (type: {entry_type}): similarity={similarity:.4f}, has_answer={candidate.answer is not None}")
                
                # Check similarity threshold
                if similarity < self.similarity_threshold:
                    logger.info(f"Candidate {candidate.id}: similarity {similarity:.4f} < threshold {self.similarity_threshold} - REJECTED")
                    continue
                
                # Example entries without answer: can be used for SQL adaptation if similarity >= 0.8
                # But cannot be used as direct cache (need to adapt SQL first)
                if entry_type == 'example' and not candidate.answer:
                    # If similarity >= 0.8 and has SQL, we can try SQL adaptation
                    if similarity >= 0.8 and candidate.sql_query:
                        logger.info(f"Candidate {candidate.id}: example entry without answer, but similarity >= 0.8 and has SQL - will try adaptation")
                        # Continue to parameter matching/adaptation logic (don't reject yet)
                    else:
                        logger.info(f"Candidate {candidate.id}: example entry without answer - REJECTED (similarity {similarity:.4f} < 0.8 or no SQL)")
                        continue
                # Cache entries without answer should also be rejected (unless we can adapt)
                elif entry_type == 'cache' and not candidate.answer:
                    # For cache entries, if similarity >= 0.8 and has SQL, try adaptation
                    if similarity >= 0.8 and candidate.sql_query:
                        logger.info(f"Candidate {candidate.id}: cache entry without answer, but similarity >= 0.8 and has SQL - will try adaptation")
                        # Continue to parameter matching/adaptation logic
                    else:
                        logger.info(f"Candidate {candidate.id}: cache entry without answer - REJECTED")
                        continue
                
                # Extract parameters from cached entry SQL
                cached_sql = candidate.sql_query
                cached_params = {}
                
                # First try to get params from metadata (if stored)
                if candidate.metadata:
                    try:
                        metadata_dict = candidate.metadata if isinstance(candidate.metadata, dict) else json.loads(candidate.metadata) if isinstance(candidate.metadata, str) else {}
                        if 'params' in metadata_dict and metadata_dict['params']:
                            cached_params = metadata_dict['params']
                            logger.info(f"Candidate {candidate.id}: using params from metadata: {cached_params}")
                    except Exception as e:
                        logger.warning(f"Candidate {candidate.id}: error parsing metadata: {e}")
                
                # If not in metadata, extract from SQL
                if not cached_params and cached_sql:
                    try:
                        cached_params = extract_sql_parameters(cached_sql)
                        logger.info(f"Candidate {candidate.id}: extracted cached params from SQL: {cached_params}")
                    except Exception as e:
                        logger.warning(f"Candidate {candidate.id}: failed to extract params from cached SQL: {e}")
                        # If extraction fails, reject this candidate
                        continue
                
                logger.info(f"Candidate {candidate.id}: cached_params={cached_params}, current_params={current_params}, sql_query_available={sql_query is not None}")
                
                # Apply safety rules
                # Rule 4: If parameter extraction FAILS on SQL that has WHERE clauses → REJECT
                if current_params is None:
                    # Current SQL extraction failed - only allow cache entries with NO parameters
                    if cached_params:
                        logger.info(f"Candidate {candidate.id}: current SQL extraction failed, rejecting parameterized cache (cached has params: {cached_params})")
                        continue
                    # Both have no params - OK
                    # But if it's an example without answer, we can't return it directly
                    if entry_type == 'example' and not candidate.answer:
                        logger.info(f"Candidate {candidate.id}: example entry without answer - cannot return directly, but similarity >= 0.8, will try SQL adaptation if SQL available")
                        if cached_sql:
                            # Try to execute the SQL directly (no adaptation needed if no params)
                            adapted_result = self._try_adapt_and_execute_sql(
                                cached_sql=cached_sql,
                                cached_params={},
                                current_params={},
                                question=question,
                                question_type=question_type,
                                similarity=similarity
                            )
                            if adapted_result:
                                logger.info(f"✅ SQL EXECUTION SUCCESS - Candidate {candidate.id} (similarity: {similarity:.4f})")
                                self._update_cache_usage(candidate.id)
                                return adapted_result
                        continue
                    logger.info(f"✅ Cache HIT - Candidate {candidate.id} (similarity: {similarity:.4f}, no params)")
                    # Update usage statistics
                    self._update_cache_usage(candidate.id)
                    return self._format_cached_result(candidate, similarity)
                
                # Rule 5: If current_sql is NOT available → Use question text extraction as fallback
                if not sql_query:
                    # We extracted params from question text (fallback)
                    # Now check if they match cached params
                    if parameters_match(current_params, cached_params):
                        # If it's an example without answer, we can't return it directly - need to execute SQL
                        if entry_type == 'example' and not candidate.answer:
                            logger.info(f"Candidate {candidate.id}: example entry without answer but params match - will execute SQL")
                            if cached_sql:
                                adapted_result = self._try_adapt_and_execute_sql(
                                    cached_sql=cached_sql,
                                    cached_params=cached_params,
                                    current_params=current_params,
                                    question=question,
                                    question_type=question_type,
                                    similarity=similarity
                                )
                                if adapted_result:
                                    logger.info(f"✅ SQL EXECUTION SUCCESS - Candidate {candidate.id} (similarity: {similarity:.4f})")
                                    self._update_cache_usage(candidate.id)
                                    return adapted_result
                            continue
                        logger.info(f"✅ Cache HIT - Candidate {candidate.id} (similarity: {similarity:.4f}, params match from question text)")
                        logger.info(f"  📝 Matched params: {current_params}")
                        # Update usage statistics
                        self._update_cache_usage(candidate.id)
                        return self._format_cached_result(candidate, similarity)
                    elif cached_params and not current_params:
                        # Cache has params but question text extraction found none
                        logger.info(f"Candidate {candidate.id}: REJECTED - cache has params {cached_params} but question text has none")
                        logger.info(f"  ⚠️  Parameter mismatch - would return wrong answer")
                        continue
                    elif current_params and not cached_params:
                        # Question has params but cache doesn't
                        logger.info(f"Candidate {candidate.id}: REJECTED - question has params {current_params} but cache has none")
                        logger.info(f"  ⚠️  Parameter mismatch - would return wrong answer")
                        continue
                    elif current_params and cached_params:
                        # Both have params but they don't match
                        # NEW FEATURE: If similarity >= 0.8, try to adapt SQL and execute without LLM
                        if similarity >= 0.8 and cached_sql:
                            logger.info(f"🔄 Candidate {candidate.id}: similarity {similarity:.4f} >= 0.8 but params don't match - attempting SQL adaptation")
                            logger.info(f"  📊 Question params: {current_params}")
                            logger.info(f"  📊 Cache params: {cached_params}")
                            
                            # Try to adapt SQL and execute
                            adapted_result = self._try_adapt_and_execute_sql(
                                cached_sql=cached_sql,
                                cached_params=cached_params,
                                current_params=current_params,
                                question=question,
                                question_type=question_type,
                                similarity=similarity
                            )
                            
                            if adapted_result:
                                logger.info(f"✅ SQL ADAPTATION SUCCESS - Candidate {candidate.id} (similarity: {similarity:.4f})")
                                # Update usage statistics
                                self._update_cache_usage(candidate.id)
                                return adapted_result
                            else:
                                logger.info(f"❌ SQL ADAPTATION FAILED - Candidate {candidate.id}, continuing to next candidate")
                                continue
                        else:
                            logger.info(f"Candidate {candidate.id}: REJECTED - parameters don't match")
                            logger.info(f"  📊 Question params: {current_params}")
                            logger.info(f"  📊 Cache params: {cached_params}")
                            logger.info(f"  ⚠️  Different parameters = different answer")
                            continue
                    else:
                        # Both have no params - OK
                        logger.info(f"✅ Cache HIT - Candidate {candidate.id} (similarity: {similarity:.4f}, no params)")
                        # Update usage statistics
                        self._update_cache_usage(candidate.id)
                        return self._format_cached_result(candidate, similarity)
                
                # Both have SQL - check parameter matching
                if parameters_match(current_params, cached_params):
                    # If it's an example without answer, we can't return it directly - need to execute SQL
                    if entry_type == 'example' and not candidate.answer:
                        logger.info(f"Candidate {candidate.id}: example entry without answer but params match - will execute SQL")
                        if cached_sql:
                            adapted_result = self._try_adapt_and_execute_sql(
                                cached_sql=cached_sql,
                                cached_params=cached_params,
                                current_params=current_params,
                                question=question,
                                question_type=question_type,
                                similarity=similarity
                            )
                            if adapted_result:
                                logger.info(f"✅ SQL EXECUTION SUCCESS - Candidate {candidate.id} (similarity: {similarity:.4f})")
                                self._update_cache_usage(candidate.id)
                                return adapted_result
                        continue
                    logger.info(f"✅ Cache HIT - Candidate {candidate.id} (similarity: {similarity:.4f}, params match: {current_params})")
                    # Update usage statistics
                    self._update_cache_usage(candidate.id)
                    return self._format_cached_result(candidate, similarity)
                else:
                    # NEW FEATURE: If similarity >= 0.8, try to adapt SQL and execute without LLM
                    if similarity >= 0.8 and cached_sql and current_params and cached_params:
                        logger.info(f"🔄 Candidate {candidate.id}: similarity {similarity:.4f} >= 0.8 but params don't match - attempting SQL adaptation")
                        logger.info(f"  📊 Current request params: {current_params}")
                        logger.info(f"  📊 Cached entry params: {cached_params}")
                        
                        # Try to adapt SQL and execute
                        adapted_result = self._try_adapt_and_execute_sql(
                            cached_sql=cached_sql,
                            cached_params=cached_params,
                            current_params=current_params,
                            question=question,
                            question_type=question_type,
                            similarity=similarity
                        )
                        
                        if adapted_result:
                            logger.info(f"✅ SQL ADAPTATION SUCCESS - Candidate {candidate.id} (similarity: {similarity:.4f})")
                            # Update usage statistics
                            self._update_cache_usage(candidate.id)
                            return adapted_result
                        else:
                            logger.info(f"❌ SQL ADAPTATION FAILED - Candidate {candidate.id}, continuing to next candidate")
                            continue
                    else:
                        logger.info(f"Candidate {candidate.id}: REJECTED - parameters don't match")
                        logger.info(f"  📊 Current request params: {current_params}")
                        logger.info(f"  📊 Cached entry params: {cached_params}")
                        logger.info(f"  ⚠️  Different parameters = different answer (e.g., different device_id)")
                        continue
            
            # No valid match found
            logger.info("No valid cache match found (similarity or parameter mismatch)")
            return None
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            import traceback
            traceback.print_exc()
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
                    csv_url = f"http://localhost:3009{csv_download_link}" if not csv_download_link.startswith('http') else csv_download_link
                    return f"The device has a total of {total_journeys} facility-to-facility movements. To access the detailed journey timing information for all these movements, you can download the complete data in CSV format from the following link: {csv_url}. This link provides the full data for the {total_journeys} journeys."
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
                        # Extract CSV link and row count
                        csv_match = re.search(r'CSV Download Link:\s*(/download-csv/[^\s\n]+)', raw)
                        row_match = re.search(r'Total rows:\s*(\d+)', raw)
                        if csv_match and row_match:
                            csv_url = f"http://localhost:3009{csv_match.group(1)}"
                            row_count = row_match.group(1)
                            return f"Found {row_count} result(s). You can download the complete data in CSV format from: {csv_url}."
                    
                    # Check if it's an empty result
                    if '0 rows' in raw or 'no rows' in raw.lower():
                        return "No results found matching your criteria."
                    
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
    
    def _update_cache_usage(self, cache_id: int):
        """
        Update usage statistics for a cache entry.
        
        Args:
            cache_id: Cache entry ID
        """
        try:
            update_query = text("""
                UPDATE ai_vector_examples
                SET usage_count = usage_count + 1,
                    last_used = CURRENT_TIMESTAMP
                WHERE id = :cache_id
            """)
            
            with self.engine.connect() as conn:
                conn.execute(update_query, {"cache_id": cache_id})
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update cache usage stats: {e}")
    
    def _format_cached_result(self, candidate, similarity: float) -> Dict[str, Any]:
        """
        Format cached result for return.
        
        Args:
            candidate: Database row with cached answer
            similarity: Similarity score
            
        Returns:
            Formatted dictionary with cached answer data
        """
        # Parse metadata
        metadata = {}
        if candidate.metadata:
            if isinstance(candidate.metadata, dict):
                metadata = candidate.metadata.copy()
            elif isinstance(candidate.metadata, str):
                try:
                    metadata = json.loads(candidate.metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
        
        # Parse result_data from metadata if available
        result_data = None
        if metadata and 'result_data' in metadata:
            result_data = metadata['result_data']
        
        return {
            "answer": candidate.answer,
            "sql_query": candidate.sql_query,
            "result_data": result_data,
            "similarity": similarity,
            "original_question": candidate.question,
            "cache_id": candidate.id
        }
    
    def save_answer_to_cache(
        self,
        question: str,
        answer: str,
        sql_query: Optional[str],
        result_data: Optional[Dict[str, Any]],
        question_type: str,
        user_type: str
    ) -> Optional[int]:
        """
        Save answer to cache with extracted parameters.
        
        Args:
            question: User's question
            answer: LLM-generated answer
            sql_query: SQL query used (if available)
            result_data: Query result data (if available)
            question_type: 'journey' or 'non_journey'
            user_type: 'admin' or 'user'
            
        Returns:
            Cache entry ID if saved, None otherwise
        """
        try:
            # Only save if deterministic (per settings)
            if settings.VECTOR_CACHE_DETERMINISTIC_ONLY:
                # For now, assume all answers are deterministic if they have SQL
                # This can be enhanced later with more sophisticated checks
                if not sql_query:
                    logger.debug("Not saving to cache: no SQL query (non-deterministic)")
                    return None
            
            # Generate embedding for the question
            query_embedding = self.vector_store.embed_query(question)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Extract parameters from SQL
            params = {}
            if sql_query:
                try:
                    params = extract_sql_parameters(sql_query)
                    logger.debug(f"Extracted parameters for cache: {params}")
                except Exception as e:
                    logger.warning(f"Failed to extract parameters from SQL: {e}")
                    # Continue anyway - params will be empty
            
            # Prepare metadata with parameters and result_data
            metadata = {
                "params": params
            }
            
            # Add result_data to metadata if available
            if result_data:
                metadata["result_data"] = result_data
            
            # Insert into cache
            insert_query = text("""
                INSERT INTO ai_vector_examples (
                    question,
                    sql_query,
                    answer,
                    question_type,
                    user_type,
                    metadata,
                    entry_type,
                    similarity_threshold,
                    is_active,
                    is_deterministic,
                    {embedding_field}
                )
                VALUES (
                    :question,
                    :sql_query,
                    :answer,
                    :question_type,
                    :user_type,
                    CAST(:metadata AS jsonb),
                    'cache',
                    :similarity_threshold,
                    true,
                    true,
                    '{embedding_str}'::vector
                )
                RETURNING id
            """.format(
                embedding_field=self.embedding_field_name,
                embedding_str=embedding_str
            ))
            
            with self.engine.connect() as conn:
                result = conn.execute(insert_query, {
                    "question": question,
                    "sql_query": sql_query or "",
                    "answer": answer,
                    "question_type": question_type,
                    "user_type": user_type,
                    "metadata": json.dumps(metadata),
                    "similarity_threshold": self.similarity_threshold
                })
                conn.commit()
                cache_id = result.fetchone()[0]
            
            logger.info(f"✅ Answer saved to cache (ID: {cache_id}, params: {params})")
            return cache_id
            
        except Exception as e:
            # Truncate error message if it contains large SQL/embeddings
            error_str = str(e)
            # Remove large embedding vectors from error messages
            if "'::vector" in error_str or "[SQL:" in error_str:
                # Find and truncate the vector/SQL portion
                if "[SQL:" in error_str:
                    sql_start = error_str.find("[SQL:")
                    if sql_start > 0:
                        error_str = error_str[:sql_start] + "[SQL: ... (truncated)]"
                elif "'::vector" in error_str:
                    # Truncate vector portion
                    vector_start = error_str.find("'[")
                    if vector_start > 0:
                        error_str = error_str[:vector_start + 50] + "... (vector truncated) ...'::vector"
            
            # Final truncation if still too long
            if len(error_str) > 300:
                error_str = error_str[:300] + "... (truncated)"
            
            logger.error(f"Error saving to cache: {error_str}")
            # Don't print full traceback for permission errors (common and expected)
            if "permission denied" not in str(e).lower() and "insufficientprivilege" not in str(e).lower():
                import traceback
                traceback.print_exc()
            return None
