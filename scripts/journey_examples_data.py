"""
Journey Calculation Examples for Vector Store

These examples demonstrate how to use the journey calculation tools.
The SQL queries fetch raw geofencing rows - journey calculation happens in Python.

IMPORTANT:
- SQL should ONLY fetch: device_id, facility_id, entry_event_time, exit_event_time
- Order by entry_event_time ASC
- Always join user_device_assignment to filter by user_id
- Journey calculation is done by journey_list_tool or journey_count_tool in Python
"""

JOURNEY_EXAMPLES = [
    {
        "question": "How many journeys from manufacturer to retailer (M to R) did device WT01C90451A36819 make in the last 180 days?",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
WHERE uda.user_id = 27
    AND dg.device_id = 'WT01C90451A36819'
    AND dg.entry_event_time >= NOW() - INTERVAL '180 days'
ORDER BY dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows for a specific device to calculate M to R journey counts. Use journey_count_tool with this SQL. The tool will calculate journeys in Python based on facility type transitions.",
        "metadata": {
            "keywords": ["journey", "manufacturer", "retailer", "M to R", "device", "count"],
            "type": "journey_count",
            "complexity": "medium",
            "tool": "journey_count_tool"
        }
    },
    {
        "question": "Count all journeys between facilities for my devices in the last 3 days",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
WHERE uda.user_id = 27
    AND dg.entry_event_time >= NOW() - INTERVAL '3 days'
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches all geofencing rows for user's devices to calculate total journey counts. Use journey_count_tool. The tool will group by device and calculate all facility-to-facility journeys.",
        "metadata": {
            "keywords": ["journey", "count", "facilities", "all devices"],
            "type": "journey_count",
            "complexity": "medium",
            "tool": "journey_count_tool"
        }
    },
    {
        "question": "List all journeys made by device WT01F3C333542F02 from facility MREFZ00004 to any other facility",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    f.facility_name,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = 27
    AND dg.device_id = 'WT01F3C333542F02'
ORDER BY dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows for a specific device to list all journeys. Use journey_list_tool. The tool will calculate journey list showing facility transitions with details.",
        "metadata": {
            "keywords": ["journey", "list", "device", "facility", "movement"],
            "type": "journey_list",
            "complexity": "medium",
            "tool": "journey_list_tool"
        }
    },
    {
        "question": "Show me all journeys from facility type M to facility type R for my devices in the last 90 days",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
WHERE uda.user_id = 27
    AND dg.entry_event_time >= NOW() - INTERVAL '90 days'
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to find M to R journeys. Use journey_list_tool. The tool will filter and show only manufacturer to retailer journeys in the results.",
        "metadata": {
            "keywords": ["journey", "manufacturer", "retailer", "M to R", "facility type"],
            "type": "journey_list",
            "complexity": "medium",
            "tool": "journey_list_tool"
        }
    },
    {
        "question": "How many total journeys did all my devices make in the last 7 days?",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
WHERE uda.user_id = 27
    AND dg.entry_event_time >= NOW() - INTERVAL '7 days'
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows for all user devices to count total journeys. Use journey_count_tool. The tool will calculate total journey count across all devices.",
        "metadata": {
            "keywords": ["journey", "count", "total", "all devices", "last 7 days"],
            "type": "journey_count",
            "complexity": "low",
            "tool": "journey_count_tool"
        }
    },
    {
        "question": "List all facility-to-facility movements for device WT01D5D2667C982E with journey times",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    f.facility_name,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = 27
    AND dg.device_id = 'WT01D5D2667C982E'
ORDER BY dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to list all journeys for a device with timing information. Use journey_list_tool. The tool will calculate journey times between facilities.",
        "metadata": {
            "keywords": ["journey", "list", "movement", "facility", "time", "device"],
            "type": "journey_list",
            "complexity": "medium",
            "tool": "journey_list_tool"
        }
    },
    {
        "question": "Count journeys from facility DALBZ00003 to any other facility for my devices",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
WHERE uda.user_id = 27
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to count journeys starting from a specific facility. Use journey_count_tool. The tool will filter and count only journeys starting from DALBZ00003.",
        "metadata": {
            "keywords": ["journey", "count", "facility", "from facility"],
            "type": "journey_count",
            "complexity": "medium",
            "tool": "journey_count_tool"
        }
    },
    {
        "question": "Show me the journey path for device WT01F3C333542F02 showing all facilities it visited",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    f.facility_name,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = 27
    AND dg.device_id = 'WT01F3C333542F02'
ORDER BY dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to show complete journey path for a device. Use journey_list_tool. The tool will list all facility visits in chronological order showing the journey path.",
        "metadata": {
            "keywords": ["journey", "path", "facilities", "visited", "device"],
            "type": "journey_list",
            "complexity": "medium",
            "tool": "journey_list_tool"
        }
    },
    {
        "question": "How many times did my devices travel from retailer to manufacturer (R to M) in the last 60 days?",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
WHERE uda.user_id = 27
    AND dg.entry_event_time >= NOW() - INTERVAL '60 days'
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to count R to M journeys. Use journey_count_tool. The tool will calculate and count only retailer to manufacturer journeys.",
        "metadata": {
            "keywords": ["journey", "retailer", "manufacturer", "R to M", "count"],
            "type": "journey_count",
            "complexity": "medium",
            "tool": "journey_count_tool"
        }
    },
    {
        "question": "List all journeys between facilities for devices that entered facility MREFZ00004 in the last month",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    f.facility_name,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = 27
    AND dg.entry_event_time >= NOW() - INTERVAL '30 days'
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows for devices that visited a specific facility. Use journey_list_tool. The tool will filter and show journeys for devices that entered the specified facility.",
        "metadata": {
            "keywords": ["journey", "list", "facility", "entered", "last month"],
            "type": "journey_list",
            "complexity": "medium",
            "tool": "journey_list_tool"
        }
    },
    {
        "question": "Count the number of journeys where devices stayed at the same facility for more than 4 hours",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
WHERE uda.user_id = 27
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to count same-facility journeys (A to A). Use journey_count_tool. The tool will identify and count journeys where device returns to the same facility after 4+ hours.",
        "metadata": {
            "keywords": ["journey", "count", "same facility", "dwell", "4 hours"],
            "type": "journey_count",
            "complexity": "high",
            "tool": "journey_count_tool"
        }
    },
    {
        "question": "Show me all journeys with facility details including facility names for device WT01F3C333542F02",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    f.facility_name,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = 27
    AND dg.device_id = 'WT01F3C333542F02'
ORDER BY dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows with facility details for a device. Use journey_list_tool. The tool will return journey list with facility names and details included.",
        "metadata": {
            "keywords": ["journey", "list", "facility details", "facility name", "device"],
            "type": "journey_list",
            "complexity": "medium",
            "tool": "journey_list_tool"
        }
    },
    {
        "question": "How many journeys did device WT01D5D2667C982E make in the last year?",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda 
    ON uda.device = dg.device_id
WHERE uda.user_id = 27
  AND dg.device_id = 'WT01D5D2667C982E'
  AND dg.entry_event_time >= NOW() - INTERVAL '365 days'
ORDER BY dg.entry_event_time ASC;
""",
        "description": "Fetches geofencing rows for a device over the last year to count total journeys. Use journey_count_tool. The tool will calculate total journey count for the device.",
        "metadata": {
            "keywords": ["journey", "count", "device", "last year"],
            "type": "journey_count",
            "complexity": "low",
            "tool": "journey_count_tool"
        }
    },
    {
        "question": "List all facility transitions for my devices showing from and to facilities",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    f.facility_name,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = 27
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to show all facility transitions. Use journey_list_tool. The tool will list all journeys showing from_facility and to_facility pairs.",
        "metadata": {
            "keywords": ["journey", "list", "facility", "transition", "from to"],
            "type": "journey_list",
            "complexity": "medium",
            "tool": "journey_list_tool"
        }
    },
    {
        "question": "Count journeys between specific facility pairs: DALBZ00003 to MREFZ00004",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
WHERE uda.user_id = 27
ORDER BY dg.device_id, dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to count journeys between specific facility pairs. Use journey_count_tool. The tool will filter and count only journeys from DALBZ00003 to MREFZ00004.",
        "metadata": {
            "keywords": ["journey", "count", "facility pair", "specific facilities"],
            "type": "journey_count",
            "complexity": "medium",
            "tool": "journey_count_tool"
        }
    },
    {
        "question": "Show me the journey history for device WT01F3C333542F02 with entry and exit times for each facility",
        "sql": """SELECT 
    dg.device_id,
    dg.facility_id,
    dg.facility_type,
    f.facility_name,
    dg.entry_event_time,
    dg.exit_event_time
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id
WHERE uda.user_id = 27
    AND dg.device_id = 'WT01F3C333542F02'
ORDER BY dg.entry_event_time ASC""",
        "description": "Fetches geofencing rows to show complete journey history with timing. Use journey_list_tool. The tool will return journey list with entry and exit times for each facility visit.",
        "metadata": {
            "keywords": ["journey", "history", "entry", "exit", "time", "device"],
            "type": "journey_list",
            "complexity": "medium",
            "tool": "journey_list_tool"
        }
    }
]

# SQL template for journey queries (for reference)
JOURNEY_SQL_TEMPLATE = """
-- Template for journey queries
-- IMPORTANT: SQL should ONLY fetch raw geofencing rows
-- Journey calculation happens in Python via journey tools

SELECT 
    dg.device_id,              -- Required: Device identifier
    dg.facility_id,            -- Required: Facility identifier
    dg.facility_type,          -- Optional: Facility type (M, R, etc.)
    f.facility_name,           -- Optional: Facility name (if needed)
    dg.entry_event_time,       -- Required: Entry time (timestamp)
    dg.exit_event_time         -- Required: Exit time (timestamp)
FROM device_geofencings dg
JOIN user_device_assignment uda ON uda.device = dg.device_id
LEFT JOIN facilities f ON dg.facility_id = f.facility_id  -- Optional: for facility names
WHERE uda.user_id = :user_id   -- ALWAYS filter by user_id
    -- Add date filters if needed:
    -- AND dg.entry_event_time >= NOW() - INTERVAL 'X days'
    -- Add device filter if needed:
    -- AND dg.device_id = 'DEVICE_ID'
ORDER BY dg.device_id, dg.entry_event_time ASC  -- CRITICAL: Must order by entry_event_time ASC
"""
