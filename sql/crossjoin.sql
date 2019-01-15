
-- Recall that cross joins do not use ON or USING
-- The power of cross join is the WHERE filter
SELECT c.name AS city, l.name AS language
FROM cities  AS c        
CROSS JOIN languages  AS l
WHERE c.name LIKE 'Hyder%' ;


SELECT c.name AS country,
    region,
    life_expectancy AS life_exp
FROM countries AS c
    LEFT JOIN populations AS p
    ON c.code = p.country_code
WHERE year = 2010
ORDER BY life_exp
LIMIT 5;


