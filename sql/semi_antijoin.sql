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