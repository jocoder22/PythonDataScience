
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

