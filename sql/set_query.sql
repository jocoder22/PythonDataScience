/*
Set join : UNION, UNION ALL, INTERSECT, EXCEPT
*/

SELECT code AS country_code
FROM currencies
UNION
SELECT country_code
FROM cities
ORDER BY country_code;

-- pick specified columns from 2010 table
SELECT *
-- 2010 table will be on top
FROM  economies2010
-- which set theory clause?
UNION 
-- pick specified columns from 2015 table
SELECT *
-- 2015 table on the bottom
FROM economies2015


SELECT code, year
FROM economies
UNION ALL
SELECT country_code, year
FROM populations
ORDER BY code, year;


SELECT code, year
FROM economies
INTERSECT
SELECT country_code, year
FROM populations
ORDER BY code, year;

SELECT name
FROM countries
INTERSECT
SELECT name
FROM cities;


SELECT name
FROM cities
Except
SELECT capital
FROM countries
ORDER BY name;

SELECT capital
FROM countries
Except
SELECT name
FROM cities
ORDER BY capital;







