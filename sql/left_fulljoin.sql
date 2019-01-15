-- get the city name(and alias it), the country code,
-- the country name(and alias it), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
region, city_proper_pop
-- specify left table
FROM cities AS c1
-- specify right table and type of join
INNER JOIN countries AS c2
-- how should the tables be matched?
ON c1.country_code = c2.code
-- sort based on descending country code
ORDER BY code DESC



-- LEFT join
-- get the city name (and alias it), the country code,
-- the country name (and alias it), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
    region, city_proper_pop
-- specify left table
FROM cities AS c1
    -- specify right table and type of join
    LEFT JOIN countries AS c2
    -- how should the tables be matched?
    ON c1.country_code = c2.code
-- sort based on descending country code
ORDER BY code DESC;



/*
select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- countries on the left (alias as c)
FROM countries AS c
    -- appropriate join with languages (as l) on the right
    LEFT JOIN languages AS l
    -- give fields to match on
    ON c.code = l.code
-- sort by descending country name
ORDER BY country DESC;