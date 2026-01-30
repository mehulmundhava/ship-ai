"""
CSV Download Routes

This module provides endpoints for downloading CSV files generated from query results.
"""

import base64
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from app.utils.csv_generator import get_csv_by_id
import logging

logger = logging.getLogger("ship_rag_ai")

router = APIRouter()


@router.get("/download-csv/{csv_id}")
def download_csv(csv_id: str):
    """
    Download CSV file by ID.
    
    Args:
        csv_id: Unique CSV identifier (UUID)
        
    Returns:
        CSV file as downloadable response with appropriate headers
        
    Raises:
        HTTPException: If CSV ID not found
    """
    logger.info(f"CSV download requested for ID: {csv_id}")
    
    # Retrieve CSV data from storage
    csv_base64 = get_csv_by_id(csv_id)
    
    if not csv_base64:
        logger.warning(f"CSV not found for ID: {csv_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CSV file with ID '{csv_id}' not found. It may have expired or been deleted."
        )
    
    try:
        # Decode base64 to get CSV content
        csv_bytes = base64.b64decode(csv_base64)
        csv_content = csv_bytes.decode('utf-8')
        
        logger.info(f"CSV retrieved successfully for ID: {csv_id}, size: {len(csv_bytes)} bytes")
        
        # Return CSV file with appropriate headers
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=query_results_{csv_id}.csv",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )
        
    except Exception as e:
        logger.error(f"Error decoding CSV for ID {csv_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing CSV file: {str(e)}"
        )
