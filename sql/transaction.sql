
BEGIN TRY 
    BEGIN TRANSACTION;
        UPDATE account SET current_balance = current_balance - 50000 WHERE accountID = 98709;
        INSERT INTO transRecord VALUES (98709, -50000, GETDATE());

        UPDATE account SET current_balance = current_balance + 50000 WHERE accountID = 66709;
        INSERT INTO transRecord VALUES (66709, 50000, GETDATE());
    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    SELECT 'Rolling back the transaction';
	ROLLBACK TRANSACTION;
END CATCH