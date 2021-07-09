#!/usr/bin/env python
import logging

logging.basicConfig(level=logging.INFO)


def xtimesPower(x, n, p):
  """ Compute the x times n to power r
  
  """
  return x * n ** p

logging.info(f"The result of {3} times {4} to the {5}th power is {xtimesPower(3, 4, 5)}")

a, b, c = 3,4,5

logging.info(f"The result of {a} times {b} to the {c}th power is {xtimesPower(a, b, c)}")

# logging default format => {level]:{logger}:{message}
# 5 hierachical levels
# 1. DEBUG  (value = 10)
# 2. INFO   (value = 20)
# 3. WARNING (value = 30)
# 4  ERROR     (value = 40)
# 5. CRITICAL  (value = 50)

### configuration
# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.ERROR)
# logging.basicConfig(level=logging.CRITICAL)

### usage
# logging.debug(f"The result of {a} times {b} to the {c}th power is {xtimesPower(a, b, c)}")
# logging.info(f"The result of {a} times {b} to the {c}th power is {xtimesPower(a, b, c)}")
# logging.warning(f"The result of {a} times {b} to the {c}th power is {xtimesPower(a, b, c)}")
# logging.error(f"The result of {a} times {b} to the {c}th power is {xtimesPower(a, b, c)}")
# logging.critical(f"The result of {a} times {b} to the {c}th power is {xtimesPower(a, b, c)}")

# to send logs to a file instead of printing to the console, add file= to the config
# logging.basicConfig(level=logging.DEBUG, file="mylogging.log")
# logging.basicConfig(level=logging.INFO, file="mylogging.log")
# logging.basicConfig(level=logging.WARNING, file="mylogging.log")
# logging.basicConfig(level=logging.ERROR, file="mylogging.log")
# logging.basicConfig(level=logging.CRITICAL, file="mylogging.log")

### logging formats
# 1. asctime = when the logRecord was created => %(asctime)s
# 2. created = when the logRecord was created => %(created)f pr %(created)s
# 3. filename = filename portion of pathname => %(filename)s
# 4. funcname = Name of fuction containing the logging call => %(funcName)s
# 5. levelname = Text logging level for the message => %(levelname)s
# 6. levelno = Numeric logging level for the meassage => %(levelno)s
# 7. lineno = Source code line number where the logging was issued => %(lineno)s
# 8. message $(message)s
# 9. module = module name portion of the pathname => $(module)s
# 10. msecs = milliseconds portion of time when the log was created => $(msecs)
# 11. pathname $(pathname)s
# 12. process $(process)d
# 13. processName $(processName)s
# 14. relativeCreated = Time in milliseconds when the LogRecord was created relative to tim when the logging module was loaded => $(relativeCreated)d
# 15. thread $(thread)d
# 16. name = name of the logger used to log the call => %(name)


# creating new logger
# always use __name__ as the logger name, as it automatically takes the filename
