"""
Table Metadata Configuration

This module contains manually configured table information including:
- Table names
- Descriptions (use cases)
- Important fields

This data is used by the get_table_list tool to provide table information
to the LLM without querying the database.

To configure tables, add entries to the TABLE_METADATA list below.
Each entry should be a dictionary with:
  - name: Table name (string)
  - description: What the table is used for / use case (string)
  - important_fields: List of important field names (list of strings)
                     You can add annotations like "(PRIMARY KEY)", "(FOREIGN KEY)", etc.
"""

# Table metadata structure
# Add your table information here
TABLE_METADATA = [
    # Example entries - replace with your actual table information:
    {
        "name": "device_details_table",
        "description": "Stores device information including device IDs, status, and latest sensor readings",
        "important_fields": [
            "device_id (PRIMARY KEY)",
            "latest_sensor_id (FOREIGN KEY -> sensors.id)",
            "latest_incoming_message_id",
            "status"
        ]
    },
    {
        "name": "user_device_assignment",
        "description": "Maps users to their assigned devices for access control",
        "important_fields": [
            "user_id (PRIMARY KEY)",
            "device_id (FOREIGN KEY -> device_details_table.device_id)",
        ]
    },
]

