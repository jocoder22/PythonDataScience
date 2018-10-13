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
creat user 'josh'@'localhost' identified by 'kelly22';
grant all on House.* to 'josh'@'localhost';


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



