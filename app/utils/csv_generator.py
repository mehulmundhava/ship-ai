"""
CSV Generation Utility

This module provides utilities for generating CSV files from query results
and managing temporary file storage.
"""

import csv
import io
import base64
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger("ship_rag_ai")

# In-memory storage for CSV files (in production, use proper file storage or S3)
_csv_storage: Dict[str, str] = {}


def generate_csv_from_result(result_text: str, max_rows: int = 5) -> Optional[Dict[str, Any]]:
    """
    Generate CSV from query result text and return metadata.
    
    Args:
        result_text: Query result as formatted string from SQLDatabase
        max_rows: Maximum number of rows to include in preview (default: 5)
        
    Returns:
        Dictionary with:
        - csv_data: Base64 encoded CSV string
        - csv_id: Unique ID for retrieving the CSV
        - row_count: Total number of rows
        - preview_rows: First N rows as list of dicts
        - csv_link: Download link (for API response)
        or None if result is empty or invalid
    """
    try:
        # Parse the result text (SQLDatabase returns formatted string)
        # Format is typically: "column1 | column2 | ...\nvalue1 | value2 | ..."
        lines = result_text.strip().split('\n')
        if len(lines) < 2:
            logger.warning("Result text has insufficient lines for CSV generation")
            return None
        
        # First line is header
        headers = [col.strip() for col in lines[0].split('|')]
        if not headers:
            logger.warning("No headers found in result text")
            return None
        
        # Parse data rows
        rows = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = [val.strip() for val in line.split('|')]
            if len(values) == len(headers):
                rows.append(dict(zip(headers, values)))
        
        if not rows:
            logger.warning("No data rows found in result text")
            return None
        
        total_rows = len(rows)
        
        # Generate CSV in memory
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        csv_content = csv_buffer.getvalue()
        
        # Encode to base64
        csv_bytes = csv_content.encode('utf-8')
        csv_base64 = base64.b64encode(csv_bytes).decode('utf-8')
        
        # Generate unique ID
        csv_id = str(uuid.uuid4())
        
        # Store CSV (in production, save to file system or S3)
        _csv_storage[csv_id] = csv_base64
        
        # Get preview rows (first max_rows)
        preview_rows = rows[:max_rows]
        
        # Generate download link (for API - would be actual endpoint in production)
        csv_link = f"/download-csv/{csv_id}"
        
        logger.info(f"Generated CSV: {total_rows} rows, ID: {csv_id}")
        
        return {
            "csv_data": csv_base64,
            "csv_id": csv_id,
            "row_count": total_rows,
            "preview_rows": preview_rows,
            "csv_link": csv_link,
            "headers": headers
        }
        
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_csv_by_id(csv_id: str) -> Optional[str]:
    """
    Retrieve CSV data by ID.
    
    Args:
        csv_id: Unique CSV identifier
        
    Returns:
        Base64 encoded CSV string, or None if not found
    """
    return _csv_storage.get(csv_id)


def format_result_with_csv(result_text: str, max_preview_rows: int = 5) -> str:
    """
    Format query result with CSV generation for large results.
    
    Args:
        result_text: Query result as formatted string
        max_preview_rows: Maximum rows to show in preview (default: 5)
        
    Returns:
        Formatted string with count, preview, and CSV link if applicable
    """
    try:
        # Parse result to count rows
        lines = result_text.strip().split('\n')
        if len(lines) < 2:
            return result_text  # Return as-is if no data
        
        # Count data rows (excluding header)
        data_rows = [line for line in lines[1:] if line.strip()]
        row_count = len(data_rows)
        
        # If <= max_preview_rows, return full result
        if row_count <= max_preview_rows:
            return result_text
        
        # Generate CSV for large results
        csv_info = generate_csv_from_result(result_text, max_preview_rows)
        
        if not csv_info:
            # Fallback: return preview only
            preview_lines = lines[:max_preview_rows + 1]  # Header + preview rows
            preview_text = '\n'.join(preview_lines)
            return f"Total rows: {row_count}\n\nFirst {max_preview_rows} rows:\n{preview_text}\n\n(Full results available via CSV download)"
        
        # Format response with count, preview, and CSV link
        preview_lines = lines[:max_preview_rows + 1]
        preview_text = '\n'.join(preview_lines)
        
        formatted_result = f"""Total rows: {csv_info['row_count']}

First {max_preview_rows} rows:
{preview_text}

CSV Download Link: {csv_info['csv_link']}
CSV ID: {csv_info['csv_id']}"""
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error formatting result with CSV: {e}")
        # Fallback: return original result
        return result_text
