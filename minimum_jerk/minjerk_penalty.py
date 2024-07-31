import numpy as np
from typing import Tuple, List

from get_terms import *
from mytypes import Array, Array2D
import utils as fcs

def cal_parameter(x_0, x_f, t_0, t_f, n):
    term_inital = [cal_position_term(t_0, n),
                   cal_velocity_term(t_0, n),
                   cal_acceleration_term(t_0, n)]

    term_final = [cal_position_term(t_f, n),
                  cal_velocity_term(t_f, n),
                  cal_acceleration_term(t_f, n)]

    M = np.vstack((term_final, term_inital))
    vec = np.vstack((x_f.reshape(-1, 1), x_0.reshape(-1, 1)))
    parameter = np.dot(np.linalg.inv(M), vec).reshape(1, -1)
    return parameter

def get_terms(t, n, mode: str) -> np.ndarray:
    x = np.zeros((len(n),  6))
    for i in range(len(n)):
        if mode == 'position':
            x[i, :] = np.array(cal_position_term(t, n[i]))
        elif mode == 'velocity':
            x[i, :] = np.array(cal_velocity_term(t, n[i]))
        elif mode == 'acceleration':
            x[i, :] = np.array(cal_acceleration_term(t, n[i]))
        elif mode == 'jerk':
            x[i, :] = np.array(cal_jerk_term(t, n[i]))
        else:
            raise ValueError(f'Specified mode does not exist!')
    return x

def cal_path(x_0, x_f, t_0, t_f, dt, n):
    nr_dof    = x_0.shape[0]
    nr_point  = int( np.round((t_f-t_0)/dt) ) + 1
    parameter = np.zeros((nr_dof, 6))

    p = np.zeros((nr_dof, nr_point))
    v = np.zeros((nr_dof, nr_point))
    a = np.zeros((nr_dof, nr_point))
    j = np.zeros((nr_dof, nr_point))

    for i_dof in range(nr_dof):
        parameter[i_dof, :] = cal_parameter(x_0[i_dof, :], x_f[i_dof, :], 0, t_f-t_0, n[i_dof])

    for i_point in range(nr_point):
        p[:, i_point] = np.sum(parameter*get_terms(i_point*dt, n, 'position'), axis=1)
        v[:, i_point] = np.sum(parameter*get_terms(i_point*dt, n, 'velocity'), axis=1)
        a[:, i_point] = np.sum(parameter*get_terms(i_point*dt, n, 'acceleration'), axis=1)
        j[:, i_point] = np.sum(parameter*get_terms(i_point*dt, n, 'jerk'), axis=1)
    return (p, v, a, j)

def get_path(x_temp, v_temp, a_temp, t_temp, n, dt):
    '''
    i_dof * [position, velocity, acceleration]
    '''
    x_0_list = np.hstack((x_temp[:, 0].reshape(-1, 1), v_temp[:, 0].reshape(-1, 1), a_temp[:, 0].reshape(-1, 1)))
    x_f_list = np.hstack((x_temp[:, 1].reshape(-1, 1), v_temp[:, 1].reshape(-1, 1), a_temp[:, 1].reshape(-1, 1)))
    return cal_path(x_0_list, x_f_list, t_temp[0], t_temp[1], dt, n)

def path_planning(X: Array2D, V: Array2D, 
                  A: Array2D, T: Array, dt: float, 
                  N: Array2D) -> tuple:
    """Plan the path using minimum jerk algorithm with the
    presence of soft constraints.

    parameters:
    -----------
    X: the given postions at certain time points, in nr_dof x nr_point
    V: the given velocities at certain time points, in nr_dof x nr_point
    A: the given accelerations at certain time points, in nr_dof x nr_point
    T: the give time points, in nr_point
    dt: the time step
    N: the penalty for the position of each segment, set as 0.0 if no soft penalty, in nr_dof x (nr_point-1)
    
    returns:
    --------
    position: the trajectory of posistions, in nr_dof x nr_step
    velocity: the trajectory of velocities, in nr_dof x nr_step
    acceleration: the trajectory of accelerations, in nr_dof x nr_step
    jerk: the trajectory of jerks, in nr_dof x nr_step
    t_stamp: then time stamp
    """
    nr_dof = X.shape[0]
    nr_point = X.shape[1]
    nr_step = fcs.get_steps(T[-1]-T[0], dt) + 1
    t_stamp = np.linspace(T[0], T[-1], nr_step, endpoint=True)

    position     = np.zeros((nr_dof, nr_step))
    velocity     = np.zeros((nr_dof, nr_step))
    acceleration = np.zeros((nr_dof, nr_step))
    jerk         = np.zeros((nr_dof, nr_step))

    position[:, 0]     = X[:, 0]
    velocity[:, 0]     = V[:, 0]
    acceleration[:, 0] = A[:, 0]
    jerk[:, 0]         = np.zeros(nr_dof)

    idx_1 = 0
    idx_2 = 1

    for i_point in range(1, nr_point):
        idx_1 = idx_2
        idx_2 = int(round((T[i_point]-T[i_point-1])/dt)) + idx_1

        [p, v, a, j] = get_path(X[:, i_point-1:i_point+1], V[:, i_point-1:i_point+1], 
                                A[:, i_point-1:i_point+1], T[i_point-1:i_point+1], 
                                N[:, i_point-1], dt)

        position[:, idx_1:idx_2]     = p[:, 1:]
        velocity[:, idx_1:idx_2]     = v[:, 1:]
        acceleration[:, idx_1:idx_2] = a[:, 1:]
        jerk[:, idx_1:idx_2]         = j[:, 1:]

    return (position, velocity, acceleration, jerk, t_stamp)