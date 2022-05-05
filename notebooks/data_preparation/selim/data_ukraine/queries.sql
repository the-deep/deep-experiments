-- 1. user_names.csv
SELECT 
  id, 
  first_name, 
  last_name 
FROM 
  auth_user
--------------------------------------------------------------------------  
-- 2. analysis_frameworks.csv
SELECT 
  * 
FROM 
  analysis_framework_analysisframework
--------------------------------------------------------------------------
-- 3. entries.csv
SELECT 
  ee.*, 
  pp.title 
FROM 
  entry_entry ee 
  INNER JOIN project_project pp ON pp.id = ee.project_id 
WHERE 
  pp.title IN (
    '2022 IMAC Ukraine'
  )
--------------------------------------------------------------------------
-- 4. af_widgets.csv
SELECT 
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
      pp.title IN (
        '2022 IMAC Ukraine'
      )
  )
--------------------------------------------------------------------------
-- 5. exportdata.csv
SELECT 
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
      pp.title IN (
        '2022 IMAC Ukraine'
        
      )
  )
--------------------------------------------------------------------------
-- 6. af_exportables.csv
SELECT 
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
      pp.title IN (
        '2022 IMAC Ukraine'
      )
  )
