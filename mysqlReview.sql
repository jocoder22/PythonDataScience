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

-- create new user with password and grant new user all priviledges
grant all on House.* to 'josh'@'localhost' identified by 'kelly22';


