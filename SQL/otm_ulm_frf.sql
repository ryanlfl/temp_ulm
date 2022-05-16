SELECT
AL1.SHIPMENT_XID, 
AL1.START_TIME,
AL1.TOTAL_SHIP_UNIT_COUNT, 
AL1.TOTAL_WEIGHT, 
AL1.TOTAL_VOLUME, 
AL1.FIRST_EQUIPMENT_GROUP_GID, 
AL1.ATTRIBUTE1 AS PRINCIPAL,
(select location_name from APP_OTMDM_ODS.ODS_LOCATION 
where location_gid = AL1.dest_location_gid) as "DestName",
(select province from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "Region",
(select city from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "City",
(select zone1 from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "DestZone1",
(select zone2 from app_otmdm_ods.ods_location
where location_gid = AL1.dest_location_gid) as "DestZone2",
AL2.ATTRIBUTE3 AS "Segment"


FROM APP_OTMDM_MYS_REPORT.ODS_SHIPMENT AL1
INNER JOIN app_otmdm_ods.ods_location AL2 ON AL2.LOCATION_GID =  AL1.dest_LOCATION_GID


WHERE  AL1.ATTRIBUTE1 = 'UNILEVER' 
        AND (AL1.START_TIME BETWEEN add_months (sysdate,-25) and sysdate )
        AND AL1.DOMAIN_NAME LIKE '%LFL/TMS/MYS%' 
        AND AL1.PERSPECTIVE='S'
        AND AL2.DOMAIN_NAME LIKE '%LFL/TMS/MYS%' 
        AND AL2.ATTRIBUTE3 IN ('DT-LOCAL', 'DT-OUTSTATION', 'IMT', 'IMT-NS')