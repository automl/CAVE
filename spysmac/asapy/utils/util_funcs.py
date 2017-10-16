import numpy as np

def get_cdf_x_y(data, cutoff):
    b_x, b_y, i_s = [], [], 0
    for i, x in enumerate(np.sort(data)):
        b_x.append(x)
        if x < cutoff:
            b_y.append(float(i) /len(data))
            i_s = i
        else: 
            b_y.append(float(i_s) /len(data))
    return b_x, b_y  