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

