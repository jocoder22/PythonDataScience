
BEGIN TRAN;
	-- Mark savepoint1
	SAVE TRAN savepoint1;
	INSERT INTO customers VALUES (988888, 10000, GETDATE());

	-- Mark savepoint2
    SAVE TRAN savepoint2;
	INSERT INTO customers VALUES (988456, 10450, GETDATE());

	-- Rollback savepoint2
	ROLLBACK TRAN savepoint2;
    -- Rollback savepoint1
	ROLLBACK TRAN savepoint1 ;

	-- Mark savepoint3
	SAVE TRAN savepoint3;
	INSERT INTO customers VALUES (982345, 23087, GETDATE());
-- Commit the transaction
COMMIT TRAN;


-- Use the appropriate setting
SET XACT_ABORT On;
-- Begin the transaction
BEGIN tran; 
	UPDATE accounts set current_balance = current_balance - 2000
		WHERE current_balance > 5000000;
	IF @@ROWCOUNT < 10	
    	-- Throw the error
		THROW 50000, 'there is error!', 1;
	ELSE		
    	-- Commit the transaction
		COMMIT tran; 

