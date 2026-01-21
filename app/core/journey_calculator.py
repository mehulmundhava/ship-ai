"""
Journey Calculator Module

This module implements the journey calculation logic ported from Laravel.
Journey calculation happens in Python, NOT in SQL.

Journey Definition:
- A journey occurs when a device moves from one facility to another
- Journey time must be >= 4 hours (14400 seconds) for different facilities
- For same facility (A -> A), minimum time is 4 hours + extraJourneyTimeLimit (if provided)
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger("ship_rag_ai")


def _convert_to_unix_timestamp(timestamp_value: Any) -> Optional[float]:
    """
    Convert various timestamp formats to Unix timestamp (float).
    
    Handles:
    - datetime objects
    - timestamp strings
    - Unix timestamps (float/int)
    - None
    
    Args:
        timestamp_value: Timestamp in any format
        
    Returns:
        Unix timestamp as float, or None if conversion fails
    """
    if timestamp_value is None:
        return None
    
    # If already a number (Unix timestamp)
    if isinstance(timestamp_value, (int, float)):
        return float(timestamp_value)
    
    # If it's a datetime object
    if isinstance(timestamp_value, datetime):
        return timestamp_value.timestamp()
    
    # If it's a string, try to parse it
    if isinstance(timestamp_value, str):
        # Remove timezone info for parsing (we'll handle timezone separately)
        timestamp_clean = timestamp_value.strip()
        
        # Try parsing as ISO format first (handles timezone)
        try:
            # Handle 'Z' timezone indicator
            if timestamp_clean.endswith('Z'):
                timestamp_clean = timestamp_clean[:-1] + '+00:00'
            dt = datetime.fromisoformat(timestamp_clean)
            return dt.timestamp()
        except (ValueError, AttributeError):
            pass
        
        # Try parsing as common PostgreSQL timestamp formats
        try:
            # List of formats to try (most specific first)
            formats = [
                '%Y-%m-%d %H:%M:%S.%f',           # With microseconds
                '%Y-%m-%d %H:%M:%S',              # Without microseconds
                '%Y-%m-%dT%H:%M:%S.%f',           # ISO with microseconds
                '%Y-%m-%dT%H:%M:%S',              # ISO without microseconds
                '%Y-%m-%d %H:%M:%S.%f+00:00',     # With timezone
                '%Y-%m-%d %H:%M:%S+00:00',        # With timezone
            ]
            
            for fmt in formats:
                try:
                    # Remove timezone part if present for strptime (strptime doesn't handle timezone well)
                    timestamp_for_parse = timestamp_clean
                    if '+' in timestamp_for_parse:
                        # Remove timezone offset (e.g., "+05:30" or "+00:00")
                        timestamp_for_parse = timestamp_for_parse.split('+')[0].rstrip()
                    elif timestamp_for_parse.endswith('Z'):
                        timestamp_for_parse = timestamp_for_parse[:-1]
                    elif ' ' in timestamp_for_parse and len(timestamp_for_parse.split()) > 2:
                        # Handle cases like "2024-01-15 10:30:00 -05:00"
                        parts = timestamp_for_parse.split()
                        if len(parts) >= 2:
                            timestamp_for_parse = ' '.join(parts[:2])
                    
                    dt = datetime.strptime(timestamp_for_parse, fmt)
                    return dt.timestamp()
                except ValueError:
                    continue
        except Exception as e:
            logger.debug(f"Error parsing timestamp string: {e}")
            pass
    
    # If all else fails, try to convert to float directly
    try:
        return float(timestamp_value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert timestamp to Unix timestamp: {timestamp_value} (type: {type(timestamp_value)})")
        return None


def valid_journey_time(
    from_time: Optional[float],
    to_time: Optional[float],
    is_same: bool = False,
    extra_hours: Optional[float] = None
) -> bool:
    """
    Validate if the time between two events constitutes a valid journey.
    
    Ported from Laravel valideJourneyTime() function.
    
    Args:
        from_time: Exit time from previous facility (Unix timestamp)
        to_time: Entry time to current facility (Unix timestamp)
        is_same: Whether the journey is from same facility to same facility (A -> A)
        extra_hours: Extra hours to add to minimum time limit for same-facility journeys
        
    Returns:
        True if the time difference is valid for a journey, False otherwise
    """
    if not from_time or not to_time:
        return False
    
    # Minimum time limit: 4 hours (14400 seconds)
    min_limit = 14400
    
    # For same facility journeys, add extra hours if provided
    if is_same and extra_hours:
        min_limit += extra_hours * 3600
    
    time_diff = to_time - from_time
    
    return time_diff >= min_limit


def calculate_journey_counts(
    geofencing_rows: List[Dict[str, Any]],
    extra_journey_time_limit: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate journey counts from geofencing rows.
    
    Ported from Laravel facility_journey_counts algorithm (1:1).
    
    Algorithm:
    1. Group rows by device_id
    2. For each device, process movements chronologically
    3. Track facility visits and calculate journeys between facilities
    4. Count journeys by facility pair (facilityA||facilityB)
    
    Args:
        geofencing_rows: List of dicts with keys:
            - device_id: Device identifier
            - facility_id: Facility identifier
            - entry_event_time: Entry time (Unix timestamp)
            - exit_event_time: Exit time (Unix timestamp)
        extra_journey_time_limit: Extra hours for same-facility journey validation
        
    Returns:
        Dictionary with:
            - counts: Dict mapping "facilityA||facilityB" to count
            - total: Total number of journeys
    """
    if not geofencing_rows:
        return {"counts": {}, "total": 0}
    
    # Step 1: Group by device_id
    device_movements: Dict[str, List[Dict[str, Any]]] = {}
    for row in geofencing_rows:
        device_id = str(row.get("device_id", ""))
        if not device_id:
            continue
        
        if device_id not in device_movements:
            device_movements[device_id] = []
        
        device_movements[device_id].append(row)
    
    # Step 2: Sort each device's movements by entry_event_time
    for device_id in device_movements:
        device_movements[device_id].sort(
            key=lambda x: _convert_to_unix_timestamp(x.get("entry_event_time")) or 0
        )
    
    # Step 3: Process each device's movements
    journey_counts: Dict[str, int] = {}
    
    # Track metadata about the data processed
    metadata = {
        "total_rows": len(geofencing_rows),
        "devices_processed": len(device_movements),
        "facility_types_found": set(),
        "facilities_found": set()
    }
    
    for device_id, movements in device_movements.items():
        visits: List[Dict[str, Any]] = []
        facility_last_index: Dict[str, int] = {}
        
        for movement in movements:
            facility_id = str(movement.get("facility_id", ""))
            entry_time = _convert_to_unix_timestamp(movement.get("entry_event_time"))
            exit_time = _convert_to_unix_timestamp(movement.get("exit_event_time"))
            
            if entry_time is None:
                logger.warning(f"Invalid entry_time for device {device_id}: {movement.get('entry_event_time')}")
                continue
            
            if not facility_id:
                continue
            
            # Track metadata
            facility_type = str(movement.get("facility_type", "")).strip()
            if facility_type:
                metadata["facility_types_found"].add(facility_type)
            metadata["facilities_found"].add(facility_id)
            
            # Add current visit (include facility_type for filtering)
            current_index = len(visits)
            visits.append({
                "facility_id": facility_id,
                "facility_type": facility_type,
                "entry_time": entry_time,
                "exit_time": exit_time
            })
            
            # Check for journeys from previous facilities
            if current_index > 0:
                for prev_facility_id, last_idx in facility_last_index.items():
                    prev_visit = visits[last_idx]
                    from_time = prev_visit.get("exit_time")
                    to_time = entry_time
                    prev_facility_type = prev_visit.get("facility_type", "")
                    
                    # Check if it's same facility
                    is_same = (prev_facility_id == facility_id)
                    
                    # Validate journey time
                    if valid_journey_time(from_time, to_time, is_same, extra_journey_time_limit):
                        # Create journey key: "facilityA||facilityB"
                        journey_key = f"{prev_facility_id}||{facility_id}"
                        
                        # Increment count
                        if journey_key not in journey_counts:
                            journey_counts[journey_key] = 0
                        journey_counts[journey_key] += 1
            
            # Update facility last index
            facility_last_index[facility_id] = current_index
    
    total = sum(journey_counts.values())
    
    # Build facility type mapping for filtering
    # Create a mapping of facility_id to facility_type from the geofencing rows
    facility_type_map: Dict[str, str] = {}
    for row in geofencing_rows:
        fid = str(row.get("facility_id", ""))
        ftype = str(row.get("facility_type", "")).strip()
        if fid and ftype and fid not in facility_type_map:
            facility_type_map[fid] = ftype
    
    # Add facility type information to journey counts
    # Format: "facilityA||facilityB" -> {"count": X, "from_type": "M", "to_type": "R"}
    journey_details: Dict[str, Dict[str, Any]] = {}
    for journey_key, count in journey_counts.items():
        parts = journey_key.split("||")
        if len(parts) == 2:
            from_facility, to_facility = parts
            from_type = facility_type_map.get(from_facility, "")
            to_type = facility_type_map.get(to_facility, "")
            journey_details[journey_key] = {
                "count": count,
                "from_facility": from_facility,
                "to_facility": to_facility,
                "from_type": from_type,
                "to_type": to_type
            }
    
    result = {
        "counts": journey_counts,
        "journey_details": journey_details,  # New: includes facility type info
        "total": total,
        "metadata": {
            "total_rows_processed": metadata["total_rows"],
            "devices_processed": metadata["devices_processed"],
            "facility_types_found": sorted(list(metadata["facility_types_found"])),
            "unique_facilities_found": len(metadata["facilities_found"]),
            "facility_type_map": facility_type_map  # New: mapping for reference
        }
    }
    
    return result


def calculate_journey_list(
    geofencing_rows: List[Dict[str, Any]],
    extra_journey_time_limit: Optional[float] = None,
    from_facility: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate journey list with facility details.
    
    Similar to calculate_journey_counts but returns detailed journey information
    including device, time pairs, and facility details.
    
    Args:
        geofencing_rows: List of dicts with keys:
            - device_id: Device identifier
            - facility_id: Facility identifier
            - facility_type: Optional facility type (M, R, etc.)
            - facility_name: Optional facility name
            - entry_event_time: Entry time (Unix timestamp)
            - exit_event_time: Exit time (Unix timestamp)
        extra_journey_time_limit: Extra hours for same-facility journey validation
        
    Returns:
        Dictionary with:
            - facilities_details: Dict mapping facility_id to facility info
            - journies: List of journey dicts with:
                - from_facility: Source facility ID
                - to_facility: Destination facility ID
                - device_id: Device identifier
                - journey_time: Time difference in seconds
                - entry_time: Entry time to destination
                - exit_time: Exit time from source
    """
    if not geofencing_rows:
        return {"facilities_details": {}, "journies": []}
    
    # Step 1: Group by device_id
    device_movements: Dict[str, List[Dict[str, Any]]] = {}
    facilities_details: Dict[str, Dict[str, Any]] = {}
    
    for row in geofencing_rows:
        device_id = str(row.get("device_id", ""))
        if not device_id:
            continue
        
        if device_id not in device_movements:
            device_movements[device_id] = []
        
        device_movements[device_id].append(row)
        
        # Collect facility details
        facility_id = str(row.get("facility_id", ""))
        if facility_id and facility_id not in facilities_details:
            facilities_details[facility_id] = {
                "facility_id": facility_id,
                "facility_type": row.get("facility_type"),
                "facility_name": row.get("facility_name")
            }
    
    # Step 2: Sort each device's movements by entry_event_time
    for device_id in device_movements:
        device_movements[device_id].sort(
            key=lambda x: _convert_to_unix_timestamp(x.get("entry_event_time")) or 0
        )
    
    # Step 3: Process each device's movements
    journies: List[Dict[str, Any]] = []
    
    # Prepare from_facility filter if specified
    from_facility_str = str(from_facility).strip() if from_facility else None
    
    for device_id, movements in device_movements.items():
        visits: List[Dict[str, Any]] = []
        facility_last_index: Dict[str, int] = {}
        
        for movement in movements:
            facility_id = str(movement.get("facility_id", ""))
            entry_time = _convert_to_unix_timestamp(movement.get("entry_event_time"))
            exit_time = _convert_to_unix_timestamp(movement.get("exit_event_time"))
            
            if entry_time is None:
                logger.warning(f"Invalid entry_time for device {device_id}: {movement.get('entry_event_time')}")
                continue
            
            if not facility_id:
                continue
            
            # Add current visit
            current_index = len(visits)
            visits.append({
                "facility_id": facility_id,
                "entry_time": entry_time,
                "exit_time": exit_time
            })
            
            # Check for journeys from previous facilities
            if current_index > 0:
                # If filtering by from_facility, only check journeys from that facility
                if from_facility_str:
                    # When filtering by from_facility, only create journeys from the immediately preceding visit
                    # if it was from the target facility. This ensures we only get direct journeys from that facility.
                    prev_visit = visits[current_index - 1]
                    prev_facility_id = prev_visit.get("facility_id")
                    
                    if prev_facility_id == from_facility_str:
                        from_time = prev_visit.get("exit_time")
                        to_time = entry_time
                        
                        # Check if it's same facility
                        is_same = (prev_facility_id == facility_id)
                        
                        # Validate journey time
                        if valid_journey_time(from_time, to_time, is_same, extra_journey_time_limit):
                            # Calculate journey time
                            journey_time = to_time - from_time if from_time else None
                            
                            # Create journey record
                            journey = {
                                "from_facility": prev_facility_id,
                                "to_facility": facility_id,
                                "device_id": device_id,
                                "journey_time": journey_time,
                                "entry_time": entry_time,
                                "exit_time": from_time
                            }
                            journies.append(journey)
                else:
                    # Original algorithm: check journeys from all previous facilities
                    for prev_facility_id, last_idx in facility_last_index.items():
                        prev_visit = visits[last_idx]
                        from_time = prev_visit.get("exit_time")
                        to_time = entry_time
                        
                        # Check if it's same facility
                        is_same = (prev_facility_id == facility_id)
                        
                        # Validate journey time
                        if valid_journey_time(from_time, to_time, is_same, extra_journey_time_limit):
                            # Calculate journey time
                            journey_time = to_time - from_time if from_time else None
                            
                            # Create journey record
                            journey = {
                                "from_facility": prev_facility_id,
                                "to_facility": facility_id,
                                "device_id": device_id,
                                "journey_time": journey_time,
                                "entry_time": entry_time,
                                "exit_time": from_time
                            }
                            journies.append(journey)
            
            # Update facility last index
            facility_last_index[facility_id] = current_index
    
    # Filter journeys by from_facility if specified (additional safety check)
    # Note: This is redundant if we already filtered during calculation, but kept for safety
    if from_facility_str:
        journies = [j for j in journies if str(j.get("from_facility", "")).strip() == from_facility_str]
        logger.info(f"Filtered to {len(journies)} journeys starting from facility {from_facility_str}")
    
    return {
        "facilities_details": facilities_details,
        "journies": journies
    }
