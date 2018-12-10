from sklearn.datasets import load_digits

digits = load_digits()
print(digits.DESCR)
ddata = digits.data
dtarget = digits.target
