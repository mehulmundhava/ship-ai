"""
CSV Generation Utility

This module provides utilities for generating CSV files from query results
and managing temporary file storage.
"""

import csv
import io
import base64
import uuid
import json
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


def generate_csv_from_journey_list(journey_result: Dict[str, Any], max_preview: int = 5) -> Optional[Dict[str, Any]]:
    """
    Generate CSV from journey list result.
    
    Args:
        journey_result: Journey result dict with 'journies' and 'facilities_details'
        max_preview: Maximum number of journeys to show in preview (default: 5)
        
    Returns:
        Dictionary with CSV metadata and formatted result string, or None if no journeys
    """
    try:
        journies = journey_result.get('journies', [])
        facilities_details = journey_result.get('facilities_details', {})
        
        if not journies:
            return None
        
        total_journeys = len(journies)
        
        # If <= max_preview, return full result without CSV
        if total_journeys <= max_preview:
            return None
        
        # Prepare CSV headers
        headers = [
            'from_facility',
            'to_facility',
            'device_id',
            'journey_time_seconds',
            'entry_time',
            'exit_time',
            'from_facility_type',
            'to_facility_type',
            'from_facility_name',
            'to_facility_name'
        ]
        
        # Convert journeys to CSV rows
        csv_rows = []
        for journey in journies:
            from_fac = journey.get('from_facility', '')
            to_fac = journey.get('to_facility', '')
            
            # Get facility details
            from_fac_details = facilities_details.get(from_fac, {})
            to_fac_details = facilities_details.get(to_fac, {})
            
            row = {
                'from_facility': from_fac,
                'to_facility': to_fac,
                'device_id': journey.get('device_id', ''),
                'journey_time_seconds': journey.get('journey_time', ''),
                'entry_time': journey.get('entry_time', ''),
                'exit_time': journey.get('exit_time', ''),
                'from_facility_type': from_fac_details.get('facility_type', ''),
                'to_facility_type': to_fac_details.get('facility_type', ''),
                'from_facility_name': from_fac_details.get('facility_name', ''),
                'to_facility_name': to_fac_details.get('facility_name', '')
            }
            csv_rows.append(row)
        
        # Generate CSV in memory
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
        csv_content = csv_buffer.getvalue()
        
        # Encode to base64
        csv_bytes = csv_content.encode('utf-8')
        csv_base64 = base64.b64encode(csv_bytes).decode('utf-8')
        
        # Generate unique ID
        csv_id = str(uuid.uuid4())
        
        # Store CSV
        _csv_storage[csv_id] = csv_base64
        
        # Get preview journeys (first max_preview)
        preview_journies = journies[:max_preview]
        
        # Generate download link
        csv_link = f"/download-csv/{csv_id}"
        
        logger.info(f"Generated journey CSV: {total_journeys} journeys, ID: {csv_id}")
        
        return {
            "csv_data": csv_base64,
            "csv_id": csv_id,
            "row_count": total_journeys,
            "preview_journies": preview_journies,
            "csv_link": csv_link,
            "headers": headers
        }
        
    except Exception as e:
        logger.error(f"Error generating journey CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_journey_list_with_csv(journey_result: Dict[str, Any], max_preview: int = 5) -> str:
    """
    Format journey list result with CSV generation for large results.
    
    Args:
        journey_result: Journey result dict with 'journies' and 'facilities_details'
        max_preview: Maximum journeys to show in preview (default: 5)
        
    Returns:
        JSON string with preview and CSV link if > max_preview journeys
    """
    try:
        journies = journey_result.get('journies', [])
        total_journeys = len(journies)
        
        # If <= max_preview, return full result
        if total_journeys <= max_preview:
            return json.dumps(journey_result, indent=2, default=str)
        
        # Generate CSV for large results
        csv_info = generate_csv_from_journey_list(journey_result, max_preview)
        
        if not csv_info:
            # Fallback: return full result
            return json.dumps(journey_result, indent=2, default=str)
        
        # Create result with preview and CSV link
        preview_result = {
            "facilities_details": journey_result.get('facilities_details', {}),
            "journies": csv_info['preview_journies'],
            "total_journeys": total_journeys,
            "preview_count": max_preview,
            "csv_download_link": csv_info['csv_link'],
            "csv_id": csv_info['csv_id'],
            "note": f"Showing first {max_preview} of {total_journeys} journeys. Download full results via CSV link."
        }
        
        return json.dumps(preview_result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error formatting journey list with CSV: {e}")
        # Fallback: return original result
        return json.dumps(journey_result, indent=2, default=str)


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
