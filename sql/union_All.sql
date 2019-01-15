/*
-- Union and Union All don't filter tables like join
each UNION query must have the same number of columns
-- but only stack them on top of the other
*/

-- pick specified columns from 2010 table
    SELECT *
    -- 2010 table will be on top
    FROM economies2010
    -- which set theory clause?
UNION
    -- pick specified columns from 2015 table
    SELECT *
    -- 2015 table on the bottom
    FROM economies2015
-- order accordingly
ORDER  BY  code, year;



/*
Determine all (non-duplicated) country codes in either the cities or 
the currencies table. The result should be a table
with only one field called country_code
*/
    SELECT code   -- final result with column name code
    FROM currencies
UNION
    SELECT country_code
    FROM cities
ORDER BY code;


    SELECT country_code  -- final result with column name country_code , why?
    FROM cities
UNION
    SELECT code
    FROM currencies
ORDER BY country_code;


    SELECT code AS country_code  -- final result with column name country_code
    FROM currencies
UNION
    SELECT country_code
    FROM cities
ORDER BY country_code;


    SELECT code, year
    FROM economies
UNION ALL
    SELECT country_code, year
    FROM populations
ORDER BY code, year;