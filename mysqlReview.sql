-- connect to mysql on mysql shell using the password
\c root@localhost  
\sql    -- Switching to SQL mode... Commands end with ;
show databases;
use world;
show tables;
show columns from city;
show columns from country;

-- connect to mysql using windows command line
-- cd into mysql server bin
cd C:\Program Files\MySQL\MySQL Server 8.0\bin
mysql -uroot -ppass23

-- or msql --user=root --password=pass23

-- create new database
create database House;
show databases;

-- create new user with password and grant new user all priviledges for database House
CREATE USER 'josh'@'localhost' IDENTIFIED BY 'kelly22';
grant all on House.* to 'josh'@'localhost';


-- UPDATE mysql.user SET Password=PASSWORD('kelly2222') WHERE User='josh@localhost';
-- UPDATE users SET pass = md5('kelly22') WHERE User='josh@localhost';

-- change user password
ALTER USER 'josh'@'localhost' IDENTIFIED BY 'kelly45';


-- Drop user
DROP USER 'josh'@'localhost';


-- show users
select host, user from mysql.user;

-- log into mysql as new user, first quit as root user
\q ; -- bye
msql -ujosh -pkelly22
show databases;  -- show the house database

-- go back to root user
\quit ; -- or exit
mysql -uroot -ppass23
show databases;

-- drop database house
drop database house;

-- give information on table in database
use world;
show tables;
describe city;  -- same a show columns from city
-- Truncate and drop
-- Truncate will clear data or entries in table
-- while drop will delete the whole table
\q ;




