-- Query the right table in information_schema
SELECT table_name 
FROM information_schema.tables
-- Specify the correct table_schema value
WHERE table_schema = 'public';


-- Query the right table in information_schema to get columns
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'unilever' AND table_schema = 'public';

-- Query the first five rows of our table
Select *
From unilever
LIMIT 5 ;

