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
