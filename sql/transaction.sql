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



-- Begin the transaction
BEGIN TRANSACTION; 
	UPDATE accounts set current_balance = current_balance + 130
		WHERE current_balance < 1000;
	-- Check number of affected rows
	IF @@ROWCOUNT > 1000 
		BEGIN 
        	-- Rollback the transaction
			ROLLBACK TRANSACTION; 
			SELECT 'More accounts than expected. Rolling back'; 
		END
	ELSE
		BEGIN 
        	-- Commit the transaction
			COMMIT TRANSACTION; 
			SELECT 'Updates commited'; 
		END

BEGIN TRY
	-- Begin the transaction
	BEGIN tran;
		UPDATE accounts SET current_balance = current_balance + 200
			WHERE account_id = 10;
    	-- Check if there is a transaction
		IF @@TRANCOUNT > 0     
    		-- Commit the transaction
			COMMIT tran;
     
	SELECT * FROM accounts
    	WHERE account_id = 10;      
END TRY
BEGIN CATCH  
    SELECT 'Rolling back the transaction'; 
    -- Check if there is a transaction
    IF @@TRANCOUNT > 0    	
    	-- Rollback the transaction
        ROLLBACK tran;
END CATCH

