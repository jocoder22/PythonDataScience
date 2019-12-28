

DECLARE @productID INT = 9854443;
DECLARE @error_message1 NVARCHAR(300) =
    CONCAT('Product with ', @productID, ' does not exit in this store');

DECLARE @error_message2 NVARCHAR(300) =
    FORMATMESSAGE('Product with %d %s ', @productID, ' does not exit in this store');

EXEC sp_addmessage
    @msgnum = 56000, @severity = 18, @msgtext = 'Product with %d %s', @lang = N'us_english';

DECLARE @error_message3 NVARCHAR(600) =
    FORMATMESSAGE(56000, @productID, ' does not exit in this store');

BEGIN TRY 
    SELECT * FROM sells WHERE productID = @productID
END TRY 
BEGIN CATCH
    THROW 50000, 'No such product in store', 1;
END CATCH

IF NOT EXISTS (SELECT * FROM sells WHERE productID = @productID)
    THROW 50000, @error_message1, 1,
ELSE
    SELECT * FROM sells WHERE productID = @productID;


