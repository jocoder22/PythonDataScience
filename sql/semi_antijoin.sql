SELECT DISTINCT name
FROM languages
WHERE code IN
  (SELECT code
   FROM countries
   WHERE region = 'Middle East')
ORDER BY name;

select code, name
from countries 
where continent = 'Oceania' 
and code not in (select code from currencies);


/*

Identify the country codes that are included in either economies or currencies but 
not in populations.
Use that result to determine the names of cities in the countries that match the 
specification in the previous instruction.

*/


-- select the city name
select c1.name
-- alias the table where city name resides
from  cities AS c1
-- choose only records matching the result of multiple set theory clauses
WHERE country_code IN
(
    -- select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- get all additional (unique) values of the field from currencies AS c2  
    union
    SELECT c2.code
    FROM currencies AS c2
    -- exclude those appearing in populations AS p
    except
    SELECT p.country_code
    FROM populations AS p
);


-- Subqueries
select * 
from populations
where  life_expectancy > 
    1.15 * (select avg(life_expectancy)
            from populations
            where year = 2015)
    and year = 2015;



-- select the appropriate fields
select name, country_code, urbanarea_pop
-- from the cities table
from  cities
-- with city name in the field of capital cities
where name IN
  (select capital
   from  countries)
ORDER BY urbanarea_pop DESC;




SELECT countries.name AS country, COUNT(*) AS cities_num
FROM cities
INNER JOIN countries
ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;


-- same as above
SELECT name AS Country,
  (SELECT count(distinct name)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num DESC, country
LIMIT 9;


SELECT local_name, subquery.lang_num
FROM countries,
  (SELECT code, COUNT(*) AS lang_num
   FROM languages
   GROUP BY code) AS subquery
WHERE countries.code = subquery.code
ORDER BY lang_num DESC;




SELECT MAX(inflation_rate) AS max_inf
  FROM (
      SELECT name, continent, inflation_rate
      FROM countries
      INNER JOIN economies
      USING (code)
      WHERE year = 2015) AS subquery
GROUP BY continent;




SELECT name, continent, inflation_rate
FROM countries
INNER JOIN economies
ON countries.code = economies.code
WHERE year = 2015
    AND inflation_rate IN (
        SELECT MAX(inflation_rate) AS max_inf
        FROM (
             SELECT name, continent, inflation_rate
             FROM countries
             INNER JOIN economies
             ON countries.code = economies.code
             WHERE year = 2015) AS subquery
        GROUP BY continent);




SELECT code, inflation_rate, unemployment_rate
FROM economies
WHERE year = 2015 AND code NOT IN
  (SELECT code
   FROM countries
   WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%'))
ORDER BY inflation_rate;





SELECT DISTINCT e.code, total_investment, imports 
FROM economies AS e
LEFT JOIN countries AS c
ON (e.code = c.code
  AND e.code IN (
    SELECT l.code
    FROM languages AS l
    WHERE official = 'true'
  ) )
WHERE year = 2015 AND e.code = c.code



SELECT DISTINCT name, total_investment, imports
FROM countries AS c
LEFT JOIN economies AS e
ON (c.code = e.code
  AND c.code IN (
    SELECT l.code
    FROM languages AS l
    WHERE official = 'true'
  ) )
WHERE region = 'Central America' AND year = 2015
ORDER BY name;


-- choose fields
SELECT  continent, region, avg(fertility_rate) AS avg_fert_rate
-- left table
FROM populations AS p
-- right table
INNER JOIN countries AS c
-- join conditions
ON p.country_code = c.code
-- specific records matching a condition
WHERE year = 2015
-- aggregated for each what?
GROUP BY  region, continent
-- how should we sort?
ORDER BY avg_fert_rate;




SELECT name, country_code, city_proper_pop,	metroarea_pop,  
      (city_proper_pop / metroarea_pop * 100) AS city_perc
FROM cities
WHERE name IN
  (SELECT capital
   FROM countries
   WHERE (continent = 'Europe'
      OR continent LIKE '%America'))
     AND metroarea_pop IS not NULL
ORDER BY city_perc desc
LIMIT 10;