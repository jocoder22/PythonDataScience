CREATE PROCEDURE sells_inserto
    @storeName VARCHAR(40),
    @productID INT, 
    @productName VARCHAR(100),
    @price DECIMAL

AS

BEGIN TRY
    INSERT INTO sells (storeName, productID, productName, price)
        VALUES (@storeName, @productID, @productName, @price);
END TRY
BEGIN CATCH
    INSERT INTO errors VALUES ('Error with inserting into sells');
END CATCH