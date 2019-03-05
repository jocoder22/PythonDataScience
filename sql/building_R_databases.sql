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


-- Create a table for the generalworks entity type
CREATE TABLE generalworks
(
    firstname text,
    lastname text
);

-- Print the contents of this table
SELECT *
FROM generalworks

-- Add the department column
ALTER TABLE generalworks
ADD COLUMN department text,
ADD COLUMN shortname text,
ADD COLUMN unishort text;

-- Print the contents of this table
SELECT *
FROM generalworks

-- Rename the department column
ALTER TABLE generalworks
RENAME COLUMN  department TO projects;


-- Create a table for the generalworks entity type
CREATE TABLE metalworks
(
    productID text,
    manufacturer text,
    size integer,
    workcode char(7),
    amount text,
    price float
);

-- Add data from another table into generalworks table
-- Insert unique unilever into the new table: generalworks
INSERT INTO generalworks 
SELECT DISTINCT firstname, lastname, shortname 
FROM unilever;

-- Doublecheck the contents of generalworks
SELECT * 
FROM generalworks;

-- Insert unique unilever into the new table: metalworks
INSERT INTO metalworks 
SELECT DISTINCT productID, manufacturer, workcode, size 
FROM unilever;

-- Doublecheck the contents of generalworks
SELECT * 
FROM metalworks;

-- done with table unilever drop it
DROP TABLE unilever;


-- Calculate the total amount as amount * price
-- using cast to change the variable attribute
SELECT productID,  CAST(amount AS float) * price AS total_amount 
FROM metalworks;