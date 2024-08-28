import numpy as np 

def diagonale(a):
    return np.prod(np.diag(a))

a = np.array([[1, 3, 8], [-1, 3, 0], [-3, 9, 2 ]])
#posso anche non dichiarare a e inserire il valore di a nella funzione diagonale
print(diagonale(a))
