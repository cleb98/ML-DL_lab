import numpy as np 

def reverse(a):
    b=np.flip(a,0)
    return b

a=np.array([1,2,3])
r = reverse(a)
print(r)
