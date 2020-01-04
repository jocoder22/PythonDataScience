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


-- IF OBJECT_ID('dbo.Employees', 'U') IS NOT NULL
-- DROP TABLE dbo.Employees
-- GO

-- IF OBJECT_ID('dbo.EmployeesupdateLog', 'U') IS NOT NULL
-- DROP TABLE dbo.EmployeesupdateLog
-- GO
-- -- SELECT * FROM dbo.Employees
-- -- GO

INSERT INTO Employees(EmployeesId, [First Name],[Last Name], Age ,Gender, Location)
  VALUES(1234, 'Peter', 'Johson', 34, 'Male', 'Newark'),
        (3456, 'Jane ', 'Skagen', 27, 'Female', 'Boston'),
        (6700, 'Flom ', 'Mongy', 31, 'Female', 'Norway'),
        (9024, 'Anthony ', 'Lowrn', 25, 'Male', 'Baltimore'),
        (1478, 'John ', 'Nokia', 32, 'Male', 'Houston'),
        (2501, 'Mary ', 'Oley', 35, 'Female', 'Atlanta'),
        (5578, 'Suaan ', 'Romey', 44, 'Female', 'India'),
        (1089, 'Kate ', 'Paoul', 21, 'Female', 'Canada');