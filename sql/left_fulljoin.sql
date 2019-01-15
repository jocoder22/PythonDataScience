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



-- Select region, average gdp_percapita (alias avg_gdp)
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias c) on the left
FROM countries AS c
    -- Join with economies (alias e)
    LEFT JOIN economies AS e
    -- Match on code fields
    ON e.code = c.code
-- Focus on 2010 
WHERE year = 2010
-- Group by region
GROUP BY region;



-- Select region, average gdp_percapita (alias avg_gdp)
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias c) on the left
FROM countries AS c
    -- Join with economies (alias e)
    LEFT JOIN economies AS e
    -- Match on code fields
    ON e.code = c.code
-- Focus on 2010 
WHERE year = 2010
-- Group by region
GROUP BY region
ORDER BY avg_gdp DESC;


SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
    indep_year, languages.name AS language, percent
FROM languages
    RIGHT JOIN countries
    ON languages.code = countries.code
    RIGHT JOIN cities
    ON countries.code = cities.country_code
ORDER BY city, language;





-- FULL JOIN
-- The power of the full join is the filtering using WHERE
SELECT name AS country, code, region, basic_unit
FROM countries
    FULL JOIN currencies
USING (code) 
WHERE region = 'North America' OR region IS NULL
ORDER BY region;

