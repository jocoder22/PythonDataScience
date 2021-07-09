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
