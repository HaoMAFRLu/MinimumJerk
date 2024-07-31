"""Analytically and efficiently calculate various order data, such as position, 
velocity, acceleration, and jerk.
"""
import numba as nb
import math
from math import cos, sin, sqrt

E = math.exp(1)
m = 1.0

@nb.jit(nopython=True)
def cal_jerk_term(t, n):
    l1 = (-(1/(E**((n*t)/m)*(6*m**4*n**2))) \
        -E**((n*t)/m)/(6*m**4*n**2) \
        +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**4*n**2)) \
        +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**4*n**2) \
        -sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(16*sqrt(3)*m**4*n**2)) \
        +(3*sqrt(3)*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*m**4*n**2)) \
        +(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(16*sqrt(3)*m**4*n**2) \
        -(3*sqrt(3)*E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(16*m**4*n**2))

    l2 = (-(1/(E**((n*t)/m)*(6*m**5*n))) \
        +E**((n*t)/m)/(6*m**5*n) \
        -cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**5*n)) \
        +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**5*n) \
        -sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(16*sqrt(3)*m**5*n)) \
        +(3*sqrt(3)*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*m**5*n)) \
        -(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(16*sqrt(3)*m**5*n) \
        +(3*sqrt(3)*E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(16*m**5*n))

    l3 = (-(1/(E**((n*t)/m)*(6*m**6))) \
        -E**((n*t)/m)/(6*m**6) \
        -cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(3*m**6)) \
        -(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(3*m**6))

    p0 = (-(n**3/(E**((n*t)/m)*(6*m**3))) \
        +(E**((n*t)/m)*n**3)/(6*m**3) \
        +(n**3*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(3*m**3)) \
        -(E**((n*t)/(2*m))*n**3*cos((sqrt(3)*n*t)/(2*m)))/(3*m**3))

    v0 = (n**2/(E**((n*t)/m)*(6*m**2)) \
        +(E**((n*t)/m)*n**2)/(6*m**2) \
        -(n**2*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m**2)) \
        -(E**((n*t)/(2*m))*n**2*cos((sqrt(3)*n*t)/(2*m)))/(6*m**2 ) \
        -(n**2*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*sqrt(3)*m**2)) \
        +(3*sqrt(3)*n**2*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*m**2)) \
        +(E**((n*t)/(2*m))*n**2*sin((sqrt(3)*n*t)/(2*m)))/(16*sqrt(3)*m**2) \
        -(3*sqrt(3)*E**((n*t)/(2*m))*n**2*sin((sqrt(3)*n*t)/(2*m)))/(16*m**2))

    a0 = (-(n/(E**((n*t)/m)*(6*m))) \
        +(E**((n*t)/m)*n)/(6*m) \
        -(n*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m)) \
        +(E**((n*t)/(2*m))*n*cos((sqrt(3)*n*t)/(2*m)))/(6*m) \
        +(n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*sqrt(3)*m)) \
        -(3*sqrt(3)*n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*m)) \
        +(E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(16*sqrt(3)*m) \
        -(3*sqrt(3)*E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(16*m))

    vec = [l1, l2, l3, p0, v0, a0]
    return vec

@nb.jit(nopython=True)
def cal_acceleration_term(t, n):
    a0 = (1/6/E**((n*t)/m) \
        +(1/6)*E**((n*t)/m) \
        +((1/3)*cos((sqrt(3)*n*t)/(2*m)))/E**((n*t)/(2*m))\
        +(1/3)*E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))

    v0 = (-(n/(E**((n*t)/m)*(6*m))) + (E**((n*t)/m)*n)/(6*m) \
        - (n*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m)) \
        + (E**((n*t)/(2*m))*n*cos((sqrt(3)*n*t)/(2*m)))/(6*m) \
        - (n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(8*sqrt(3)*m)) \
        - (sqrt(3)*n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(8*m)) \
        - (E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(8*sqrt(3)*m) \
        - (sqrt(3)*E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(8*m))

    p0 = (n**2/(E**((n*t)/m)*(6*m**2)) + (E**((n*t)/m)*n**2)/(6*m**2) - \
        (n**2*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m**2)) - \
        (E**((n*t)/(2*m))*n**2*cos((sqrt(3)*n*t)/(2*m)))/(6*m**2) + \
        (n**2*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**2)) - \
        (E**((n*t)/(2*m))*n**2*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**2)) 

    l1= (1/(E**((n*t)/m)*(6*m**3*n**3)) \
        - E**((n*t)/m)/(6*m**3*n**3) \
        - cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(3*m**3*n**3)) \
        + (E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(3*m**3*n**3))

    l2 = (1/(E**((n*t)/m)*(6*m**4*n**2)) \
        + E**((n*t)/m)/(6*m**4*n**2) \
        - cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**4*n**2)) \
        - (E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**4*n**2) \
        - sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(8*sqrt(3)*m**4*n**2)) \
        - (sqrt(3)*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(8*m**4*n**2)) \
        + (E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(8*sqrt(3)*m**4*n**2) \
        + (sqrt(3)*E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(8*m**4*n**2))

    l3 = (1/(E**((n*t)/m)*(6*m**5*n)) \
        - E**((n*t)/m)/(6*m**5*n) \
        + cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**5*n)) \
        - (E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**5*n) \
        - sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**5*n)) \
        - (E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**5*n))

    vec = [l1, l2, l3, p0, v0, a0]
    return vec

@nb.jit(nopython=True)
def cal_velocity_term(t, n):
    v0 = (1/6/E**((n*t)/m) \
        +(1/6)*E**((n*t)/m) \
        +((1/3)*cos((sqrt(3)*n*t)/(2*m)))/E**((n*t)/(2*m)) \
        +(1/3)*E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))

    l2 = (-(1/(E**((n*t)/m)*(6*m**3*n**3))) \
        +E**((n*t)/m)/(6*m**3*n**3) \
        +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(3*m**3*n**3)) \
        -(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(3*m**3*n**3))

    l1 = (-(1/(E**((n*t)/m)*(6*m**2*n**4))) \
        -E**((n*t)/m)/(6*m**2*n**4) \
        +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**2*n**4)) \
        +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**2*n**4) \
        -sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**2*n**4)) \
        +(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**2*n**4))

    l3 = (-(1/(E**((n*t)/m)*(6*m**4*n**2))) \
        -E**((n*t)/m)/(6*m**4*n**2) \
        +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**4*n**2)) \
        +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**4*n**2) \
        +sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**4*n**2)) \
        -(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**4*n**2))

    a0 = (-(m/(E**((n*t)/m)*(6*n))) \
        +(E**((n*t)/m)*m)/(6*n) \
        -(m*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*n)) \
        +(E**((n*t)/(2*m))*m*cos((sqrt(3)*n*t)/(2*m)))/(6*n) \
        +(m*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*n)) \
        +(E**((n*t)/(2*m))*m*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*n))

    p0 = (-(n/(E**((n*t)/m)*(6*m))) \
        +(E**((n*t)/m)*n)/(6*m) \
        -(n*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m)) \
        +(E**((n*t)/(2*m))*n*cos((sqrt(3)*n*t)/(2*m)))/(6*m) \
        -(n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*m)) \
        -(E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m))

    vec = [l1, l2, l3, p0, v0, a0]
    return vec

@nb.jit(nopython=True)
def cal_position_term(t, n):
    a0 = (m**2/(E**((n*t)/m)*(6*n**2)) \
        +(E**((n*t)/(2*m))*m**2*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*n**2) \
        +(E**((n*t)/m)*m**2)/(6*n**2) \
        -(E**((n*t)/(2*m))*m**2*cos((sqrt(3)*n*t)/(2*m)))/(6*n**2) \
        -(m**2*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*n**2)) \
        -(m**2*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*n**2)))

    l3 = (-(cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(3*m**3*n**3))) \
        +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(3*m**3*n**3) \
        -E**((n*t)/m)/(6*m**3*n**3)+1/(E**((n*t)/m)*(6*m**3*n**3)))

    l2 = (-(cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**2*n**4))) \
        +sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**2*n**4)) \
        +E**((n*t)/m)/(6*m**2*n**4)-(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**2*n**4) \
        -(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**2*n**4) \
        +1/(E**((n*t)/m)*(6*m**2*n**4)))

    l1 = (cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m*n**5)) \
        +sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m*n**5)) \
        +(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m*n**5) \
        -E**((n*t)/m)/(6*m*n**5)-(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m*n**5) \
        +1/(E**((n*t)/m)*(6*m*n**5)))

    v0 = (-(m/(E**((n*t)/m)*(6*n))) \
        +(E**((n*t)/(2*m))*m*cos((sqrt(3)*n*t)/(2*m)))/(6*n) \
        +(m*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*n)) \
        +(E**((n*t)/(2*m))*m*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*n) \
        +(E**((n*t)/m)*m)/(6*n)-(m*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*n)))

    p0 = ((1/3)*E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)) \
        +(1/6)*E**((n*t)/m) \
        +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*3) \
        +1/(6*E**((n*t)/m)))

    vec = [l1, l2, l3, p0, v0, a0]
    return vec

