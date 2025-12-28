"""
Example Data for Vector Store

This module contains all the example queries and extra prompt data
that were previously embedded in the long prompt. These are stored
as arrays and will be loaded into FAISS vector store for retrieval.
"""

# Example SQL queries with their corresponding questions
# These will be used for few-shot learning via RAG retrieval
SAMPLE_EXAMPLES = [
    {
        "question": "Can you provide the number of 'manufacturer to retailer' journeys done by asset ID WT01F3C333542F02 in last 180 days between M to R for user 27?",
        "sql": """WITH journey_data AS (
                SELECT
                    dg.device_id,
                    dg.facility_type,
                    dg.entry_event_time,
                    LAG(dg.facility_type) OVER (PARTITION BY dg.device_id ORDER BY dg.entry_event_time) AS prev_facility_type
                FROM device_geofencings dg
                JOIN user_device_assignment uda ON uda.device = dg.device_id
                WHERE uda.user_id = 27
                AND dg.device_id = 'WT01F3C333542F02'
                AND dg.entry_event_time >= EXTRACT(EPOCH FROM (NOW() - INTERVAL '180 days'))
                ORDER BY dg.entry_event_time
            ),
            journey_transitions AS (
                SELECT
                    device_id,
                    facility_type,
                    entry_event_time,
                    CASE
                        WHEN prev_facility_type = 'M' AND facility_type = 'R' THEN 1
                        ELSE 0
                    END AS is_m_to_r_journey
                FROM journey_data
            )
            SELECT COUNT(*) AS manufacturer_to_retailer_journeys
            FROM journey_transitions
            WHERE is_m_to_r_journey = 1;""",
                    "metadata": {
                        "keywords": [
                            "journey",
                            "manufacturer",
                            "retailer",
                            "M to R",
                            "device",
                            "facility",
                            "geofencing",
                        ],
                        "type": "journey_count",
                        "complexity": "high",
                    },
                },
    {
        "question": "Count devices that have current temperature more than 10 degree C",
        "sql": """SELECT COUNT(*) as device_count
            FROM device_current_data cd
            JOIN user_device_assignment ud ON cd.device_id = ud.device
            WHERE ud.user_id = '66' 
            AND cd.temperature > 10;""",
        "metadata": {
            "keywords": ["temperature", "device count", "current", "threshold"],
                "type": "device_count",
                "complexity": "medium",
            },
        },
    {
        "question": "Count devices that have current battery more than 80%",
        "sql": """SELECT COUNT(*) as device_count
            FROM device_current_data cd
            INNER JOIN user_device_assignment ud ON cd.device_id = ud.device
            WHERE ud.user_id = '66' AND cd.battery > 80""",
        "metadata": {
            "keywords": ["battery", "device count", "current", "percentage"],
            "type": "device_count",
            "complexity": "medium",
        },
    },
    {
        "question": "find devices list that are currently in 'MREFZ00004' facility",
        "sql": """SELECT cd.device_id, cd.device_name
            FROM device_current_data cd
            JOIN user_device_assignment ud ON cd.device_id = ud.device
            WHERE ud.user_id = '66' AND cd.facility_id = 'MREFZ00004';""",
        "metadata": {
            "keywords": ["facility", "device list", "location", "current"],
            "type": "device_list",
            "complexity": "low",
        },
    },
    {
        "question": "find devices list that have dwell time more than 1 day",
        "sql": """SELECT count(*)
            FROM device_current_data cd
            JOIN user_device_assignment ud ON cd.device_id = ud.device
            WHERE ud.user_id = '66' AND cd.dwell_time_seconds > 86400;""",
        "metadata": {
            "keywords": ["dwell time", "device", "duration", "seconds"],
            "type": "device_count",
            "complexity": "medium",
        },
    },
    {
        "question": "Count devices currently located in New York",
        "sql": """SELECT count(*)
            FROM device_current_data cd
            JOIN user_device_assignment ud ON ud.device = cd.device_id
            WHERE ST_Contains(
            ST_GeomFromText(
                'POLYGON((
                -74.25909 40.917577,
                -73.950000 40.800000,
                -73.700272 40.477399,
                -74.25909 40.477399,
                -74.25909 40.917577
                ))',
                4326
            ),
            ST_SetSRID(ST_MakePoint(cd.longitude, cd.latitude), 4326)
            )
            AND ud.user_id = '66';""",
        "metadata": {
            "keywords": [
                "location",
                "geography",
                "polygon",
                "latitude",
                "longitude",
                "New York",
            ],
            "type": "device_count",
            "complexity": "high",
        },
    },
    {
        "question": "Count of devices that reported data in the last 2 hours",
        "sql": """SELECT COUNT(*) AS device_count
            FROM device_current_data cd 
            JOIN user_device_assignment ud ON ud.device = cd.device_id
            WHERE cd.updated_at >= NOW() - INTERVAL '2 hours'
            AND ud.user_id = '66';""",
        "metadata": {
            "keywords": ["recent", "time interval", "hours", "data report"],
            "type": "device_count",
            "complexity": "medium",
        },
    },
    {
        "question": "count of devices that generated event in the last 2 hours",
        "sql": """SELECT COUNT(*) AS device_count
            FROM device_current_data cd
            JOIN user_device_assignment ud ON ud.device = cd.device_id
            WHERE cd.event_time >= NOW() - INTERVAL '2 hours'
            AND ud.user_id = '27';""",
        "metadata": {
            "keywords": ["event", "time interval", "hours", "recent"],
            "type": "device_count",
            "complexity": "medium",
        },
    },
    {
        "question": "count devices that generated free-fall in last 1 day",
        "sql": """SELECT COUNT(*) AS device_count
            FROM device_current_data cd
            JOIN user_device_assignment ud ON ud.device = cd.device_id
            WHERE cd.free_fall_event_time >= NOW() - INTERVAL '1 day'
            AND ud.user_id = '27';""",
        "metadata": {
            "keywords": ["free-fall", "shock", "event", "time interval", "day"],
            "type": "device_count",
            "complexity": "medium",
        },
    },
    {
        "question": "count devices that generated shock in last 1 day",
        "sql": """SELECT COUNT(*) AS device_count
            FROM device_current_data cd
            JOIN user_device_assignment ud ON ud.device = cd.device_id
            WHERE cd.shock_event_time >= NOW() - INTERVAL '1 day'
            AND ud.user_id = '27';""",
        "metadata": {
            "keywords": ["shock", "event", "time interval", "day"],
            "type": "device_count",
            "complexity": "medium",
        },
    },
]

# Extra prompt data for business rules and table descriptions
# These provide context about database schema and business logic
EXTRA_PROMPT_DATA = [
    {
        "content": "device_current_data (CD) - Contains current data of devices like temperature, battery, dwell-time, event-time(location event-time), free-fall-event-time, shock-event-time, facility-id, facility-type. Join field: CD.device_id = D.device_id and ud.device = cd.device_id ",
        "metadata": {
            "keywords": ["device_current_data", "temperature", "battery", "dwell-time", "event-time", "free-fall-event-time", "shock-event-time", "facility-id", "facility-type"],
            "type": "schema_info",
        },
    },
    {
        "content": "Journey is defined - When device goes from one facility id to another facilityid that's known as facility. A journey for a particular device ID is identified when the facility ID changes between a specified before time and after time. If the facility type also changes during this transition, the journey is classified as an X to Y journey, where X represents the facility type before the change and Y represents the facility type after the change.",
        "metadata": {
            "keywords": [
                "journey",
                "facility",
                "device",
                "transition",
                "facility type",
            ],
            "type": "business_rule",
        },
    },
    {
        "content": "Facility type LHS to RHS conversion: manufacturer -> M, retailer -> R. Facility type columns has values like M, R, U, D, etc.",
        "metadata": {
            "keywords": [
                "facility type",
                "manufacturer",
                "retailer",
                "conversion",
                "M",
                "R",
            ],
            "type": "business_rule",
        },
    },
    {
        "content": "admin (A) - This is user list table. To fetch data for any device, always check if user role_id is 2 or 3 (normal-user) then look in user_device_assignment Table to check if device belongs to login user or not. Field role_id values: 1 = super-admin (have all device and table access), 2,3 = users (only have access of specific device - check device and user_id field in user_device_assignment table).",
        "metadata": {
            "keywords": ["admin", "user", "role", "access control", "permission"],
            "type": "schema_info",
        },
    },
    {
        "content": "device_details_table (D) - Contains details of devices like device-name, grai id, IMEI number. Also have current data id of other table. Latest_incoming_message_id: latest SNo of incoming_message_history_k table for location(lat-long), facility, temperature, battery, dwell-time, event-time. Latest_shock_id: latest id of sensor table for shock event. Latest_free_fall_id: latest id of sensor table for free-fall event. Latest_sensor_id: latest id of sensor table for temperature and battery. Latest_light_id: latest id of light_data table for light-data event.",
        "metadata": {
            "keywords": [
                "device_details_table",
                "device",
                "IMEI",
                "latest",
                "sensor",
                "location",
            ],
            "type": "schema_info",
        },
    },
    {
        "content": "user_device_assignment (UD) - Use for Get list of devices that user has assign. If login user role is 2 or 3, then always join this table with particular target table like incoming_message_history_K, sensor, shock_info, device_alerts, device_geofencings tables with where condition of user_device_assignment.user_id and user_device_assignment.device field. Join field: UD.device = D.Device_ID, A.id = UD.user_id",
        "metadata": {
            "keywords": [
                "user_device_assignment",
                "join",
                "user_id",
                "device",
                "access control",
            ],
            "type": "schema_info",
        },
    },
    {
        "content": "incoming_message_history_K (IK) - Use for get Location history (lat-long) of devices. It also contain dwell-time(how long device is there in same location), move mark (first entry at which device has change it's location). Join field: IK.device_id = D.Device_ID, IK.facility_id = F.facility_id",
        "metadata": {
            "keywords": [
                "incoming_message_history_k",
                "location",
                "latitude",
                "longitude",
                "dwell",
                "facility",
            ],
            "type": "schema_info",
        },
    },
    {
        "content": "facilities (F) - Premises in which device may take-hold for some time. While fetching facility data compare company_id field with login user id. Join field: IK.facility_id = F.facility_id, F.company_id = A.id",
        "metadata": {
            "keywords": ["facilities", "premises", "company_id", "join"],
            "type": "schema_info",
        },
    },
    {
        "content": "sensor (S) - Have some sensor data history for devices like temperature and battery. Join field: S.device_id = D.Device_ID",
        "metadata": {
            "keywords": ["sensor", "temperature", "battery", "history", "device"],
            "type": "schema_info",
        },
    },
    {
        "content": "shock_info - Have some stroke data history for devices like shock and free-fall. Join field: S.device_id = D.Device_ID (type = shock, free-fall)",
        "metadata": {
            "keywords": ["shock_info", "shock", "free-fall", "event", "history"],
            "type": "schema_info",
        },
    },
    {
        "content": "device_geofencings (DG) - Give data about Movement of device in facility (entry-exit time). Join: DG.device_id = D.Device_ID && F.facility_id = DB.facility_id",
        "metadata": {
            "keywords": ["device_geofencings", "movement", "entry", "exit", "facility"],
            "type": "schema_info",
        },
    },
    {
        "content": "device_alerts - Give last sensor values that cross it's threshold limit",
        "metadata": {
            "keywords": ["device_alerts", "threshold", "sensor", "alert"],
            "type": "schema_info",
        },
    },
    {
        "content": "Dwell time is stored in seconds inside a field named dwell_timestamp. 1 day = 24 hours × 60 minutes × 60 seconds = 86400 seconds.",
        "metadata": {
            "keywords": ["dwell", "time", "seconds", "timestamp", "conversion"],
            "type": "business_rule",
        },
    },
]
