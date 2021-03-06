SELECT
S.SHIPMENT_GID,
S.SHIPMENT_XID, 
S.SERVPROV_GID,
S.TOTAL_SHIP_UNIT_COUNT, 
S.TOTAL_WEIGHT, 
S.TOTAL_VOLUME, 
S.FIRST_EQUIPMENT_GROUP_GID, 
S.EQUIPMENT_REFERENCE_UNIT_GID,
S.USER_DEFINED1_ICON_GID,
S.ATTRIBUTE4 AS "ATTRIBUTE4(DRIVER_ID)",
S.ATTRIBUTE5 AS "ATTRIBUTE5(VEHICLE_NO)",
S.TOTAL_ACTUAL_COST, 
S.LOADED_DISTANCE, 
S.DEST_LOCATION_GID, 
S.ATTRIBUTE1 AS PRINCIPAL,
S.ATTRIBUTE2 AS "ATTRIBUTE2(ROUTE)",
S.TRANSPORT_MODE_GID,
S.TERM_LOCATION_TEXT,
N.location_name as "SourceName",
L.location_name as "DestName",
L.province AS "Region",
L.city as "City",
L.zone1 as "DestZone1",
L.zone2 as "DestZone2",
L.ATTRIBUTE3 AS "Segment",
S.START_TIME,
S.END_TIME
FROM Blue.V_OTM_ODS_SHIPMENT AS S
LEFT OUTER JOIN Blue.V_OTM_ODS_LOCATION AS L ON S.dest_location_gid = L.LOCATION_GID
LEFT OUTER JOIN Blue.V_OTM_ODS_LOCATION AS N ON S.dest_location_gid = N.LOCATION_GID
WHERE  S.ATTRIBUTE1 = 'UNILEVER' 
   --AND S.START_TIME >= DATEADD(month, -25, GETDATE())
   AND S.START_TIME >= DATEADD(month, -1, GETDATE())
   AND S.DOMAIN_NAME = 'LFL/TMS/MYS' 
   AND S.PERSPECTIVE='S'