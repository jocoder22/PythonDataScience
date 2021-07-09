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
# logging.debug()
# logging.info()
# logging.warning()
# logging.error()
# logging.critical()
