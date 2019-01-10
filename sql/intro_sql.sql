
SELECT * 
FROM people
WHERE firstname like '_R%' ;

SELECT  firstname , language_Used
FROM films
WHERE language_Used IN ('French', 'Spanish');

SELECT title, release_year
FROM films
WHERE (release_years > 1990 AND release_year < 2000)	
AND (budget > 1000000 AND language IN ('French', 'Latin'));

-- this counts the number of rows in the table
SELECT count(*)
FROM films;

-- this count the number of unique job titles for women
SELECT count( DISTINCT jobs)
FROM employee
WHERE gender = 'F';


-- Aggregating functions
SELECT avg(budget)
FROM films;

SELECT min(age)
FROM people;

SELECT sum(allowance)
FROM finace;

SELECT max(gross)
FROM films
WHERE release_year BETWEEN 2000 AND 2012;


SELECT release_year, country, min(gross)
FROM films
GROUP BY release_year , country
ORDER BY country, release_year;

SELECT release_year, avg(budget) AS avg_budget, avg(gross) AS avg_gross
FROM films
WHERE release_year > 1990
GROUP BY release_year
HAVING avg(budget) > 60000000
ORDER BY avg_gross DESC;