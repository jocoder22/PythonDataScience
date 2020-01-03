-- Create a new database called 'TutorialDB'
-- Connect to the 'master' database to run this snippet
-- F1 localhost master localhostmaster
-- https://docs.microsoft.com/en-us/sql/visual-studio-code/sql-server-develop-use-vscode?view=sql-server-ver15
USE master
GO
IF NOT EXISTS (
   SELECT name
   FROM sys.databases
   WHERE name = N'TutorialDB'
)
CREATE DATABASE [TutorialDB]
GO

-- use ctrl+shift+E

-- Create a new table called 'Employes' in schema 'dbo'
-- Drop the table if it already exists
IF OBJECT_ID('dbo.Employees', 'U') IS NOT NULL
DROP TABLE dbo.Employees
GO
-- Create the table in the specified schema
CREATE TABLE dbo.Employees
(
    EmployeesId INT NOT NULL PRIMARY KEY, -- primary key column
    [First Name] [NVARCHAR](50) NOT NULL,
    [Last Name] [NVARCHAR](50) NOT NULL,
    Age INT NOT NULL,
    Gender [NVARCHAR](50) NOT NULL,
    Location [NVARCHAR](50) NOT NULL
    -- specify more columns here
);
GO

-- Insert rows into table 'Employees'
INSERT INTO Employees
   ([EmployeesId],[First Name],[Last Name], Age, Gender,[Location])
VALUES
   ( 1123, N'Jared', N'Mondey', 45, N'Male', N'Australia'),
   ( 2456, N'Nikita', N'Nonra', 35, N'Female', N'India'),
   ( 3789, N'Tom', N'Godde', 40, N'Male', N'Canada'),
   ( 4895, N'Jake', N'Roney', 24, N'Male', N'Ukraine'),
   ( 1789, N'Rom', N'Frankline', 30, N'Male', N'Germany'),
   ( 7895, N'Makre', N'Venna', 26, N'Female', N'United States'),
   ( 5789, N'Vincent', N'Peters', 38, N'Male', N'Italy'),
   ( 6895, N'Paul', N'Marlag', 29, N'Male', N'United States')
GO
-- Query the total count of employees
SELECT COUNT(*) as EmployeeCount FROM dbo.Employees;
-- Query all employee information
SELECT e.EmployeesId, e.[First Name], e.[Last Name], e.Age, e.Gender, e.Location
FROM dbo.Employees as e
GO

IF OBJECT_ID('dbo.Employees', 'U') IS NOT NULL
DROP TABLE dbo.Employees
GO