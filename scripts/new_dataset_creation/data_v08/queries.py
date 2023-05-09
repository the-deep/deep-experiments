analysis_framework_q = """
SELECT 
  * 
FROM 
  analysis_framework_analysisframework
"""

entries_q = """SELECT 
  ee.*,
  pp.id as prj_id,
  pp.title
FROM 
  entry_entry ee 
  INNER JOIN project_project pp ON pp.id = ee.project_id 
WHERE 
  pp.id IN (
    '{}'
  )
"""

af_widget_q = """SELECT 
  * 
FROM 
  analysis_framework_widget 
WHERE 
  analysis_framework_id IN (
    SELECT 
      DISTINCT ee.analysis_framework_id 
    FROM 
      entry_entry ee 
      INNER JOIN analysis_framework_analysisframework af ON af.id = ee.analysis_framework_id 
      INNER JOIN project_project pp ON pp.id = ee.project_id 
    WHERE 
      pp.id IN (
        '{}'
      )
  )
"""

exportdata_q = """SELECT 
  * 
FROM 
  entry_exportdata 
WHERE 
  entry_exportdata.entry_id IN (
    SELECT 
      ee.id 
    FROM 
      entry_entry ee 
      INNER JOIN project_project pp ON pp.id = ee.project_id 
    WHERE 
      pp.id IN (
        '{}'
        
      )
  )
"""

af_exportables_q = """SELECT 
  * 
FROM 
  analysis_framework_exportable 
WHERE 
  analysis_framework_id IN (
    SELECT 
      DISTINCT ee.analysis_framework_id 
    FROM 
      entry_entry ee 
      INNER JOIN analysis_framework_analysisframework af ON af.id = ee.analysis_framework_id 
      INNER JOIN project_project pp ON pp.id = ee.project_id 
    WHERE 
      pp.id IN (
        '{}'
      )
  )
"""

projects_q = """SELECT 
  * 
FROM 
  project_project pp 
WHERE 
  pp.id IN (
    '{}'
  )
"""

lead_q = """SELECT 
  *
FROM 
  lead_lead ll
WHERE 
  ll.project_id IN (
    '{}'
  )
"""

geolocation_q = """SELECT 
id, title
FROM
  geo_geoarea
"""
