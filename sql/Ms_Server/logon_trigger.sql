-- Create trigger to log logins
IF NOT EXISTS ( SELECT  name
            FROM    sys.server_triggers
            WHERE   type = 'TR'
                    AND name = 'logonTrigger' ) 
BEGIN
    EXEC('CREATE TRIGGER logonTrigger
ON ALL SERVER WITH EXECUTE AS ''sa''
FOR LOGON
As 
    INSERT INTO LogonLog(LogonName, LogonDate, SessionID, SourceIPAddress)
    SELECT ORIGINAL_LOGIN(), GETDATE(), @@spid, client_net_address
    FROM sys.dm_exec_connections WHERE session_id = @@SPID');
END
GO

EXEC msdb.dbo.sp_send_dbmail  
    @profile_name = 'Adventure Works Administrator',  
    @recipients = 'okigbookey@gmail.com',  
    @body = 'The stored procedure finished successfully.',  
    @subject = 'Automated Success Message' ; 

-- DESKTOP-STF6SAO

USE [msdb]
GO
EXEC msdb.dbo.sp_set_sqlagent_properties @email_save_in_sent_folder=1, 
		@databasemail_profile=N''
GO


Configuring...

- Create new account 'SQLAlerts' for SMTP server 'DESKTOP-STF6SAO' (Success)

- Create New profile 'okeyokigbo' (Success)

- Add account 'SQLAlerts' to profile 'okeyokigbo' with priority '1' (Success)

- Grant 'guest' access to 'okeyokigbo' (Success)



-- https://www.sqlshack.com/email-sql-query-results-smartphone-using-sp_send_dbmail-stored-procedure/
-- https://www.mssqltips.com/sqlservertip/2922/sql-server-alerts-with-text-messaging-from-sql-server-database-mail/