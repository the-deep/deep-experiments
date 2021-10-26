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
    '2020 DFS Libya', '2020 DFS Nigeria', 
    'Americas Regional Population Movement', 
    'Bosnia and Herzegovina_Population Movement Report', 
    'COVID-19 Americas Region Multi-Sectorial Assessment', 
    'Central America - Dengue Outbreak 2019', 
    'Central America: Hurricanes Eta and Iota', 
    'GIMAC Afghanistan', 'GIMAC Cameroon', 
    'GIMAC Chad', 'GIMAC Niger', 'GIMAC Somalia', 
    'GIMAC South Sudan', 'GIMAC Sudan', 
    'IFRC - Cyclone Idai, March 2019', 
    'IFRC Chile', 'IFRC Democratic Republic of Congo', 
    'IFRC Guatemala', 'IFRC India', 
    'IFRC Kenya', 'IFRC Niger', 'IFRC Nigeria', 
    'IFRC Peru', 'IFRC Philippines', 
    'IFRC Tajikistan', 'IFRC Turkey', 
    'IFRC Uganda', 'IFRC Yemen Sit Analysis, July 2020', 
    'IFRC Yemen', 'IMMAP/DFS Bangladesh', 
    'IMMAP/DFS Burkina Faso', 'IMMAP/DFS Colombia', 
    'IMMAP/DFS Nigeria', 'IMMAP/DFS RDC', 
    'IMMAP/DFS Syria', 'Lebanon Situation Analysis', 
    'Libya Situation Analysis (OA)', 
    'Nigeria Situation Analysis (OA)', 
    'Situation Analysis Generic Libya', 
    'Situation Analysis Generic Yemen', 
    'Sudan Floods - September 2020', 
    'The Bahamas - Hurricane Dorian - Early Recovery Assessment', 
    'UNHCR Americas', 'UNHCR Argentina', 
    'UNHCR Aruba', 'UNHCR Bolivia', 
    'UNHCR Chile', 'UNHCR Colombia', 
    'UNHCR Costa Rica', 'UNHCR Curacao', 
    'UNHCR Dominican Republic', 'UNHCR Ecuador', 
    'UNHCR El Salvador', 'UNHCR Guatemala', 
    'UNHCR Guyana', 'UNHCR Honduras', 
    'UNHCR Mexico', 'UNHCR Panama', 
    'UNHCR Paraguay', 'UNHCR Peru', 
    'UNHCR Trinidad and Tobago', 'UNHCR Uruguay', 
    'UNHCR Venezuela', 'Venezuela crisis 2019', 
    'Yemen Situation Analysis (OA)'
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
        '2020 DFS Libya', '2020 DFS Nigeria', 
        'Americas Regional Population Movement', 
        'Bosnia and Herzegovina_Population Movement Report', 
        'COVID-19 Americas Region Multi-Sectorial Assessment', 
        'Central America - Dengue Outbreak 2019', 
        'Central America: Hurricanes Eta and Iota', 
        'GIMAC Afghanistan', 'GIMAC Cameroon', 
        'GIMAC Chad', 'GIMAC Niger', 'GIMAC Somalia', 
        'GIMAC South Sudan', 'GIMAC Sudan', 
        'IFRC - Cyclone Idai, March 2019', 
        'IFRC Chile', 'IFRC Democratic Republic of Congo', 
        'IFRC Guatemala', 'IFRC India', 
        'IFRC Kenya', 'IFRC Niger', 'IFRC Nigeria', 
        'IFRC Peru', 'IFRC Philippines', 
        'IFRC Tajikistan', 'IFRC Turkey', 
        'IFRC Uganda', 'IFRC Yemen Sit Analysis, July 2020', 
        'IFRC Yemen', 'IMMAP/DFS Bangladesh', 
        'IMMAP/DFS Burkina Faso', 'IMMAP/DFS Colombia', 
        'IMMAP/DFS Nigeria', 'IMMAP/DFS RDC', 
        'IMMAP/DFS Syria', 'Lebanon Situation Analysis', 
        'Libya Situation Analysis (OA)', 
        'Nigeria Situation Analysis (OA)', 
        'Situation Analysis Generic Libya', 
        'Situation Analysis Generic Yemen', 
        'Sudan Floods - September 2020', 
        'The Bahamas - Hurricane Dorian - Early Recovery Assessment', 
        'UNHCR Americas', 'UNHCR Argentina', 
        'UNHCR Aruba', 'UNHCR Bolivia', 
        'UNHCR Chile', 'UNHCR Colombia', 
        'UNHCR Costa Rica', 'UNHCR Curacao', 
        'UNHCR Dominican Republic', 'UNHCR Ecuador', 
        'UNHCR El Salvador', 'UNHCR Guatemala', 
        'UNHCR Guyana', 'UNHCR Honduras', 
        'UNHCR Mexico', 'UNHCR Panama', 
        'UNHCR Paraguay', 'UNHCR Peru', 
        'UNHCR Trinidad and Tobago', 'UNHCR Uruguay', 
        'UNHCR Venezuela', 'Venezuela crisis 2019', 
        'Yemen Situation Analysis (OA)'
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
        '2020 DFS Libya', '2020 DFS Nigeria', 
        'Americas Regional Population Movement', 
        'Bosnia and Herzegovina_Population Movement Report', 
        'COVID-19 Americas Region Multi-Sectorial Assessment', 
        'Central America - Dengue Outbreak 2019', 
        'Central America: Hurricanes Eta and Iota', 
        'GIMAC Afghanistan', 'GIMAC Cameroon', 
        'GIMAC Chad', 'GIMAC Niger', 'GIMAC Somalia', 
        'GIMAC South Sudan', 'GIMAC Sudan', 
        'IFRC - Cyclone Idai, March 2019', 
        'IFRC Chile', 'IFRC Democratic Republic of Congo', 
        'IFRC Guatemala', 'IFRC India', 
        'IFRC Kenya', 'IFRC Niger', 'IFRC Nigeria', 
        'IFRC Peru', 'IFRC Philippines', 
        'IFRC Tajikistan', 'IFRC Turkey', 
        'IFRC Uganda', 'IFRC Yemen Sit Analysis, July 2020', 
        'IFRC Yemen', 'IMMAP/DFS Bangladesh', 
        'IMMAP/DFS Burkina Faso', 'IMMAP/DFS Colombia', 
        'IMMAP/DFS Nigeria', 'IMMAP/DFS RDC', 
        'IMMAP/DFS Syria', 'Lebanon Situation Analysis', 
        'Libya Situation Analysis (OA)', 
        'Nigeria Situation Analysis (OA)', 
        'Situation Analysis Generic Libya', 
        'Situation Analysis Generic Yemen', 
        'Sudan Floods - September 2020', 
        'The Bahamas - Hurricane Dorian - Early Recovery Assessment', 
        'UNHCR Americas', 'UNHCR Argentina', 
        'UNHCR Aruba', 'UNHCR Bolivia', 
        'UNHCR Chile', 'UNHCR Colombia', 
        'UNHCR Costa Rica', 'UNHCR Curacao', 
        'UNHCR Dominican Republic', 'UNHCR Ecuador', 
        'UNHCR El Salvador', 'UNHCR Guatemala', 
        'UNHCR Guyana', 'UNHCR Honduras', 
        'UNHCR Mexico', 'UNHCR Panama', 
        'UNHCR Paraguay', 'UNHCR Peru', 
        'UNHCR Trinidad and Tobago', 'UNHCR Uruguay', 
        'UNHCR Venezuela', 'Venezuela crisis 2019', 
        'Yemen Situation Analysis (OA)'
        
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
        '2020 DFS Libya', '2020 DFS Nigeria', 
        'Americas Regional Population Movement', 
        'Bosnia and Herzegovina_Population Movement Report', 
        'COVID-19 Americas Region Multi-Sectorial Assessment', 
        'Central America - Dengue Outbreak 2019', 
        'Central America: Hurricanes Eta and Iota', 
        'GIMAC Afghanistan', 'GIMAC Cameroon', 
        'GIMAC Chad', 'GIMAC Niger', 'GIMAC Somalia', 
        'GIMAC South Sudan', 'GIMAC Sudan', 
        'IFRC - Cyclone Idai, March 2019', 
        'IFRC Chile', 'IFRC Democratic Republic of Congo', 
        'IFRC Guatemala', 'IFRC India', 
        'IFRC Kenya', 'IFRC Niger', 'IFRC Nigeria', 
        'IFRC Peru', 'IFRC Philippines', 
        'IFRC Tajikistan', 'IFRC Turkey', 
        'IFRC Uganda', 'IFRC Yemen Sit Analysis, July 2020', 
        'IFRC Yemen', 'IMMAP/DFS Bangladesh', 
        'IMMAP/DFS Burkina Faso', 'IMMAP/DFS Colombia', 
        'IMMAP/DFS Nigeria', 'IMMAP/DFS RDC', 
        'IMMAP/DFS Syria', 'Lebanon Situation Analysis', 
        'Libya Situation Analysis (OA)', 
        'Nigeria Situation Analysis (OA)', 
        'Situation Analysis Generic Libya', 
        'Situation Analysis Generic Yemen', 
        'Sudan Floods - September 2020', 
        'The Bahamas - Hurricane Dorian - Early Recovery Assessment', 
        'UNHCR Americas', 'UNHCR Argentina', 
        'UNHCR Aruba', 'UNHCR Bolivia', 
        'UNHCR Chile', 'UNHCR Colombia', 
        'UNHCR Costa Rica', 'UNHCR Curacao', 
        'UNHCR Dominican Republic', 'UNHCR Ecuador', 
        'UNHCR El Salvador', 'UNHCR Guatemala', 
        'UNHCR Guyana', 'UNHCR Honduras', 
        'UNHCR Mexico', 'UNHCR Panama', 
        'UNHCR Paraguay', 'UNHCR Peru', 
        'UNHCR Trinidad and Tobago', 'UNHCR Uruguay', 
        'UNHCR Venezuela', 'Venezuela crisis 2019', 
        'Yemen Situation Analysis (OA)'
      )
  )
--------------------------------------------------------------------------
-- 7. projects.csv
SELECT 
  * 
FROM 
  project_project pp 
WHERE 
  pp.title IN (
    '2020 DFS Libya', '2020 DFS Nigeria', 
    'Americas Regional Population Movement', 
    'Bosnia and Herzegovina_Population Movement Report', 
    'COVID-19 Americas Region Multi-Sectorial Assessment', 
    'Central America - Dengue Outbreak 2019', 
    'Central America: Hurricanes Eta and Iota', 
    'GIMAC Afghanistan', 'GIMAC Cameroon', 
    'GIMAC Chad', 'GIMAC Niger', 'GIMAC Somalia', 
    'GIMAC South Sudan', 'GIMAC Sudan', 
    'IFRC - Cyclone Idai, March 2019', 
    'IFRC Chile', 'IFRC Democratic Republic of Congo', 
    'IFRC Guatemala', 'IFRC India', 
    'IFRC Kenya', 'IFRC Niger', 'IFRC Nigeria', 
    'IFRC Peru', 'IFRC Philippines', 
    'IFRC Tajikistan', 'IFRC Turkey', 
    'IFRC Uganda', 'IFRC Yemen Sit Analysis, July 2020', 
    'IFRC Yemen', 'IMMAP/DFS Bangladesh', 
    'IMMAP/DFS Burkina Faso', 'IMMAP/DFS Colombia', 
    'IMMAP/DFS Nigeria', 'IMMAP/DFS RDC', 
    'IMMAP/DFS Syria', 'Lebanon Situation Analysis', 
    'Libya Situation Analysis (OA)', 
    'Nigeria Situation Analysis (OA)', 
    'Situation Analysis Generic Libya', 
    'Situation Analysis Generic Yemen', 
    'Sudan Floods - September 2020', 
    'The Bahamas - Hurricane Dorian - Early Recovery Assessment', 
    'UNHCR Americas', 'UNHCR Argentina', 
    'UNHCR Aruba', 'UNHCR Bolivia', 
    'UNHCR Chile', 'UNHCR Colombia', 
    'UNHCR Costa Rica', 'UNHCR Curacao', 
    'UNHCR Dominican Republic', 'UNHCR Ecuador', 
    'UNHCR El Salvador', 'UNHCR Guatemala', 
    'UNHCR Guyana', 'UNHCR Honduras', 
    'UNHCR Mexico', 'UNHCR Panama', 
    'UNHCR Paraguay', 'UNHCR Peru', 
    'UNHCR Trinidad and Tobago', 'UNHCR Uruguay', 
    'UNHCR Venezuela', 'Venezuela crisis 2019', 
    'Yemen Situation Analysis (OA)'
  )