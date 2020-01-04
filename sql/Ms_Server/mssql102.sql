SELECT Country, AVG(life) AS AverageLife
FROM dbo.gapminder
GROUP BY Country
Having Country like 'Un%'
ORDER BY AverageLife DESC