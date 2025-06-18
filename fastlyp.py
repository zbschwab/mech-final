import numpy as np
from numba import njit

test = "flyp says hi!"  # module link test

# initial conditions — damped driven pendulum (both linear and quadratic)
omega0 = 3*np.pi  # natural frequency
omega = 2*np.pi  # driving frequency
beta = 3*np.pi/4
phi_0 = 1
dphi_0 = 0
t0 = 0.
dt = 1e-2

r_init_ddp = np.array([phi_0, dphi_0], dtype=np.float64)

# initial conditions - double pendulum
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
theta1_0 = np.pi/2
theta2_0 = 0.0
dtheta1_0 = 0.0
dtheta2_0 = 0.0
g = 9.81

r_init_dp = np.array([theta1_0, theta2_0, dtheta1_0, dtheta2_0], dtype=np.float64)


# system function - linearly-damped driven pendulum
@njit
def f_lddp(r_now, t, gamma):
    # unpack the variables
    res = np.zeros_like(r_now)
    theta = r_now[0]
    angvel = r_now[1]

    # calculate the derivatives
    res[0] = angvel
    res[1] = gamma*omega0**2 * np.cos(omega*t) - 2*beta*angvel - omega0**2 * np.sin(theta)

    return res


# system function - quadratically-damped driven pendulum
@njit
def f_qddp(r_now, t, gamma):
    # unpack the variables
    res = np.zeros_like(r_now)
    theta = r_now[0]
    angvel = r_now[1]

    # calculate the derivatives
    res[0] = angvel
    res[1] = gamma*omega0**2 * np.cos(omega*t) - 2*beta*np.sign(angvel)*(angvel**2) - omega0**2 * np.sin(theta)

    return res


# system function - double pendulum
@njit
def f_dp(r_now, t):
    # unpack the variables
    res = np.zeros_like(r_now)
    theta1 = r_now[0]
    theta2 = r_now[1]
    dtheta1 = r_now[2]
    dtheta2 = r_now[3]

    # calculate components of the derivatives (for convenience/legibility)
    diff = theta1 - theta2
    f_1 = -(l2/l1) * (m2/(m1+m2)) * (dtheta2**2) * np.sin(diff) * (g/l1) * np.sin(theta1)
    f_2 = -(l1/l2) * (dtheta1**2) * np.sin(diff) * (g/l2) * np.sin(theta2)
    alpha_1 = (l2/l1) * (m2/(m1+m2)) * np.cos(diff)
    alpha_2 = (l1/l2) * np.cos(diff)
    denom = 1 - (alpha_1*f_2)

    # calculate the derivatives
    res[0] = dtheta1
    res[1] = dtheta2
    res[2] = (f_1 - alpha_1*f_2) / denom
    res[3] = (f_2 - alpha_2*f_1) / denom

    return res


# jacobian - linearly-damped driven pendulum
@njit
def jac_lddp(r_now, t, param=None):
    res = np.zeros((r_now.shape[0], r_now.shape[0]))
    res[0,0] = 0
    res[0,1] = 1
    res[1,0] = -omega0**2 * np.cos(r_now[0])
    res[1,1] = -2 * beta

    return res


# jacobian - quadratically-damped driven pendulum
@njit
def jac_qddp(r_now, t, param=None):
    res = np.zeros((r_now.shape[0], r_now.shape[0]))
    res[0,0] = 0
    res[0,1] = 1
    res[1,0] = -omega0**2 * np.cos(r_now[0])
    res[1,1] = -4 * beta * np.sign(r_now[1]) * r_now[1]

    return res


# jacobian - double pendulum
@njit
def jac_dp(r_now, t):
    # unpack the variables
    res = np.zeros((r_now.shape[0], r_now.shape[0]))
    theta1 = r_now[0]
    theta2 = r_now[1]
    dtheta1 = r_now[2]
    dtheta2 = r_now[3]
    diff = theta1 - theta2 #Define diff for convenience

    #Define the Jacobian
    res[0,0] = 0
    res[0,1] = 0
    res[0,2] = 1
    res[0,3] = 0

    res[1,0] = 0
    res[1,1] = 0
    res[1,2] = 0
    res[1,3] = 1

    res[2,0] = (-2*(g*m1*np.cos(theta1)+m2*l1*(dtheta1**2)*np.cos(2*diff)+m2*np.cos(diff)*(l2*(dtheta2**2)+g*np.cos(theta2)))*(m1+m2*(np.sin(diff)**2))+m2*(g*(2*m1+m2)*np.sin(theta1)+g*m2*np.sin(theta1-2*theta2)+2*m2*(l2*(dtheta2**2)+l1*(dtheta1**2)*np.cos(diff))*np.sin(diff))*np.sin(2*diff))/(2*l1*(m1+m2*(np.sin(diff)**2))**2)
    res[2,1] = (2*m2*(-m2*(l1*(dtheta1**2)+g*np.cos(theta1))+(2*m1-m2)*l2*(dtheta2**2)*np.cos(diff)+((2*m1+m2)*(l1*(dtheta1**2)+g*np.cos(dtheta1))+m2*l2*(dtheta2**2)*np.cos(diff))*np.cos(2*diff)))/(l1*(2*m1+m2-m2*np.cos(2*diff))**2)
    res[2,2] = (-m2*dtheta1*np.sin(2*diff))/(m1+m2*(np.sin(diff)**2))
    res[2,3] = (-2*m2*l2*dtheta2*np.sin(diff))/(l1*(m1+m2*(np.sin(diff)**2)))

    res[3,0] = (-2*(m1+m2)*(l1*(dtheta1**2)+g*np.cos(theta1))*np.cos(diff)*(m1-m2*(np.sin(diff)**2))+2*(-m2*l2*(dtheta2**2)*np.cos(2*diff)+g*(m1+m2)*np.sin(theta1)*np.sin(diff))*(m1+m2*(np.sin(diff)**2)+(m2**2)*l2*(dtheta2**2)*(np.sin(2*diff)**2)))/(-2*l2*((m1+m2*(np.sin(diff)**2))**2))
    res[3,1] = (-1*((m1+m2)*(l1*(dtheta1**2)+g*np.cos(theta1))*np.cos(diff)*(2*m1-m2+m1*np.cos(2*diff)))+m2*l2*(dtheta2**2)*(m2-(2*m1+m2)*np.cos(2*diff)))/(2*l2*((m1+m2*(np.sin(diff)**2))**2))
    res[3,2] = (2*(m1+m2)*l1*dtheta1*np.sin(diff))/(l2*(m1+m2*(np.sin(diff)**2)))
    res[3,3] = (m2*dtheta2*np.sin(2*diff))/(m1+m2*(np.sin(diff)**2))

    return res


# compute state x of system after one dt w/ RK4 method - lddp
# (simulates trajectory for dt)
@njit
def rk4_step_lddp(x, t, dt, param=None):
    k1 = f_lddp(x, t, param)
    k2 = f_lddp(x + 0.5*dt*k1, t + 0.5*dt, param)
    k3 = f_lddp(x + 0.5*dt*k2, t + 0.5*dt, param)
    k4 = f_lddp(x + dt*k3, t + dt, param)
    x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_next


# compute state x of system after one dt w/ RK4 method - qddp
@njit
def rk4_step_qddp(x, t, dt, param=None):
    k1 = f_qddp(x, t, param)
    k2 = f_qddp(x + 0.5*dt*k1, t + 0.5*dt, param)
    k3 = f_qddp(x + 0.5*dt*k2, t + 0.5*dt, param)
    k4 = f_qddp(x + dt*k3, t + dt, param)
    x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_next


# compute state x of system after one dt w/ RK4 method - dp
@njit
def rk4_step_dp(x, t, dt):
    k1 = f_dp(x, t)
    k2 = f_dp(x + 0.5*dt*k1, t + 0.5*dt)
    k3 = f_dp(x + 0.5*dt*k2, t + 0.5*dt)
    k4 = f_dp(x + dt*k3, t + dt)
    x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_next


# compute state D of deviation vector after one dt w/ RK4 method - lddp
# (simulates whether trajectory diverges or converges for dt)
@njit
def rk4_LTM_step_lddp(x, t, dt, D, param=None):
    jac = jac_lddp(x, t, param)
    k1 = jac @ D
    k2 = jac @ (D + 0.5*dt*k1)
    k3 = jac @ (D + 0.5*dt*k2)
    k4 = jac @ (D + dt*k3)
    D_next = D + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return D_next


# compute state D of deviation vector after one dt w/ RK4 method - qddp
@njit
def rk4_LTM_step_qddp(x, t, dt, D, param=None):
    jac = jac_qddp(x, t, param)
    k1 = jac @ D
    k2 = jac @ (D + 0.5*dt*k1)
    k3 = jac @ (D + 0.5*dt*k2)
    k4 = jac @ (D + dt*k3)
    D_next = D + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return D_next


# compute state D of deviation vector after one dt w/ RK4 method - dp
@njit
def rk4_LTM_step_dp(x, t, dt, D):
    jac = jac_dp(x, t)
    k1 = jac @ D
    k2 = jac @ (D + 0.5*dt*k1)
    k3 = jac @ (D + 0.5*dt*k2)
    k4 = jac @ (D + dt*k3)
    D_next = D + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return D_next


# calculate max lyapunov constant for system at given parameter - lddp
@njit
def mLCE_core_lddp(x, t, dt, n_forward, n_compute, param):
    dim = x.shape[0]

    # discard transients for better lyapunov solving
    for _ in range(n_forward):
        x = rk4_step_lddp(x, t, dt, param)
        t += dt

    # set initial deviation vector
    d = np.random.rand(dim)
    d /= np.linalg.norm(d)

    mLCE = 0.0

    for i in range(1, n_compute + 1):
        d = rk4_LTM_step_lddp(x, t, dt, d, param)
        x = rk4_step_lddp(x, t, dt, param)
        t += dt

        norm_d = np.linalg.norm(d)
        mLCE += np.log(norm_d)
        d /= norm_d

    mLCE_next = mLCE / (n_compute * dt)
    return mLCE_next


# calculate max lyapunov constant for system at given parameter - qddp
@njit
def mLCE_core_qddp(x, t, dt, n_forward, n_compute, param):
    dim = x.shape[0]

    # discard transients for better lyapunov solving
    for _ in range(n_forward):
        x = rk4_step_qddp(x, t, dt, param)
        t += dt

    # set initial deviation vector
    d = np.random.rand(dim)
    d /= np.linalg.norm(d)

    mLCE = 0.0

    for i in range(1, n_compute + 1):
        d = rk4_LTM_step_qddp(x, t, dt, d, param)
        x = rk4_step_qddp(x, t, dt, param)
        t += dt

        norm_d = np.linalg.norm(d)
        mLCE += np.log(norm_d)
        d /= norm_d

    mLCE_next = mLCE / (n_compute * dt)
    return mLCE_next


# calculate max lyapunov constant for system at given parameter - dp
@njit
def mLCE_core_dp(x, t, dt, n_forward, n_compute):
    dim = x.shape[0]

    # discard transients for better lyapunov solving
    for _ in range(n_forward):
        x = rk4_step_dp(x, t, dt)
        t += dt

    # set initial deviation vector
    d = np.random.rand(dim)
    d /= np.linalg.norm(d)

    mLCE = 0.0

    for i in range(1, n_compute + 1):
        d = rk4_LTM_step_dp(x, t, dt, d)
        x = rk4_step_dp(x, t, dt)
        t += dt

        norm_d = np.linalg.norm(d)
        mLCE += np.log(norm_d)
        d /= norm_d

    mLCE_next = mLCE / (n_compute * dt)
    return mLCE_next


n_forward = 0  # how much of the initial oscillation to discard
n_compute = 10**6  # how many steps to calculate max lyapunov over


# wrap mLCE_core to take one parameter (allows multiprocessing) - lddp
def mLCE_lddp(param):
    x0 = r_init_ddp.copy()
    lyp = mLCE_core_lddp(x0, t0, dt, n_forward, n_compute, param)
    return lyp


# wrap mLCE_core to take one parameter (allows multiprocessing) - qddp
def mLCE_qddp(param):
    x0 = r_init_ddp.copy()
    lyp = mLCE_core_qddp(x0, t0, dt, n_forward, n_compute, param)
    return lyp


# wrap mLCE_core to take one parameter (allows multiprocessing) - dp
def mLCE_dp(param):
    r_init_dp[0] = param
    x0 = r_init_dp.copy()
    lyp = mLCE_core_dp(x0, t0, dt, n_forward, n_compute)
    return lyp