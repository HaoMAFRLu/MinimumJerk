import math
import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'minimum_jerk')))
from minjerk import minimum_jerk_trajectory

def test():
    dt = 0.01  # step size
    # penalty for all DoFs
    n_list = [0.1, 10.0, 0.1]
    n_list = np.array(n_list)
    
    x_list = [0.0, math.pi/4, 0.0, 0.0]
    x_list = np.array(x_list)
    
    v_list = [0.0, 5.0, 0.0, 0.0]
    v_list = np.array(v_list)
    
    a_list = [0.0, 0.0, 0.0, 0.0]
    a_list = np.array(a_list)

    t_list = [0.0, 1.0, 2.0, 2.2]
    t_list = np.array(t_list)
    position, velocity, acceleration, jerk, t_stamp = minimum_jerk_trajectory(x_list, v_list, a_list, t_list, dt)

    nr_dof = position.shape[0]
    legend_position = 'lower right'

    fig = plt.figure(figsize=(16, 16))
    ax_position = fig.add_subplot(411)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Angle $\theta$ in degree')
    line = []
    for i in range(nr_dof):
        line_temp, = ax_position.plot(t_stamp, position[i, :] * 180 / math.pi, linewidth=2, label=r'Pos. dof {}'.format(i+1))
        line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)
        
    ax_velocity = fig.add_subplot(412)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Velocity $v$ in rad/s')
    line = []
    for i in range( nr_dof ):
        line_temp, = ax_velocity.plot(t_stamp, velocity[i, :], linewidth=2, label=r'Vel. dof {}'.format(i+1))
        line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)

    ax_acceleration = fig.add_subplot(413)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Acceleration $a$ in rad/$s^2$')
    line = []
    for i in range( nr_dof ):
        line_temp, = ax_acceleration.plot(t_stamp, acceleration[i, :], linewidth=2, label=r'Acc. dof {}'.format(i+1))
        line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)

    ax_jerk = fig.add_subplot(414)
    plt.xlabel(r'Time $t$ in s')
    plt.ylabel(r'Jerk $j$ in rad/$s^3$')
    line = []
    for i in range( nr_dof ):
        line_temp, = ax_jerk.plot(t_stamp, jerk[i, :], linewidth=2, label=r'Jerk. dof {}'.format(i+1))
        line.append( line_temp )
    plt.legend(handles=line, loc=legend_position, shadow=True)

    plt.show()

if __name__ == '__main__':
    test()
    