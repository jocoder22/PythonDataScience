DECLARE @productID INT = 9854443

BEGIN TRY 
    SELECT * FROM sells WHERE productID = @productID
END TRY 
BEGIN CATCH
    THROW 50034, 'No such product in store', 1;
END CATCH

IF NOT EXISTS (SELECT * FROM sells WHERE productID = @productID)
    THROW 50034, 'No such product in store', 1,
ELSE
    SELECT * FROM sells WHERE productID = @productID;


