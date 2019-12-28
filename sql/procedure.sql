CREATE PROCEDURE sells_inserto
    @storeName VARCHAR(100),
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


BEGIN TRY
    EXEC sells_inserto
        @storeName = 'NewYork345',
        @productID = 980234,
        @productName = "Samsung HDF Television",
        @price = 589.99
END TRY
BEGIN CATCH
    SELECT ERROR_MESSAGE();
END CATCH
