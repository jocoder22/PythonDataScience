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
IF OBJECT_ID('dbo.TablesChangeLog', 'U') IS NOT NULL
DROP TABLE dbo.TablesChangeLog
GO


-- Create the table in the specified schema
IF NOT EXISTS (SELECT name FROM sysobjects WHERE name='TablesChangeLog' and xtype='U')
    CREATE TABLE TablesChangeLog(
      EventData NVARCHAR(MAX) NOT NULL, 
      ServerName NVARCHAR(MAX) NOT NULL, 
      Object NVARCHAR(MAX) NOT NULL, 
      ChangedBy NVARCHAR(MAX) NOT NULL,
      Query NVARCHAR(MAX) NOT NULL,
      ChangeDate DATE NOT NULL
    )
GO

ALTER TABLE TablesChangeLog   
  ALTER COLUMN EventData xml;
ALTER TABLE TablesChangeLog 
  ALTER COLUMN ChangedBy xml;
ALTER TABLE TablesChangeLog 
  ALTER COLUMN ChangeDate DATETIME;


IF NOT EXISTS (SELECT name FROM sysobjects WHERE name='Employees' and xtype='U')
      CREATE TABLE dbo.Employees(
         EmployeesId INT NOT NULL, -- primary key column
         [First Name] [NVARCHAR](50) NOT NULL,
         [Last Name] [NVARCHAR](50) NOT NULL,
         Age INT NOT NULL,
         Gender [NVARCHAR](50) NOT NULL,
         Location [NVARCHAR](50) NOT NULL
      )
GO


-- Create the table in the specified schema
IF NOT EXISTS (SELECT name FROM sysobjects WHERE name='EmployeesupdateLog' and xtype='U')
CREATE TABLE dbo.EmployeesupdateLog
(
    EmployeesId INT NOT NULL , -- primary key column
    [First Name] [NVARCHAR](50) NOT NULL,
    [Last Name] [NVARCHAR](50) NOT NULL,
    Age INT NOT NULL,
    Gender [NVARCHAR](50) NOT NULL,
    Location [NVARCHAR](50) NOT NULL,
    DateChanged DATETIME
);
GO


-- Query the total count of employees
SELECT COUNT(*) as EmployeeCount FROM dbo.Employees;
-- Query all employee information
SELECT e.EmployeesId, e.[First Name], e.[Last Name], e.Age, e.Gender, e.Location
FROM dbo.Employees as e
GO



-- IF OBJECT_ID('dbo.Employees', 'U') IS NOT NULL
-- DROP TABLE dbo.Employees
-- GO

-- IF OBJECT_ID('dbo.EmployeesupdateLog', 'U') IS NOT NULL
-- DROP TABLE dbo.EmployeesupdateLog
-- GO
-- -- SELECT * FROM dbo.Employees
-- -- GO

