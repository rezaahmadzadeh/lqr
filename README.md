## Linear Quadratci Regulator (LQR) 

This repository provides the LQR class in python that includes:
- Time-independent LQR
- Time-varying LQR

Dependencies:
1. numpy
2. scipy
3. matplotlib (for plotting results)


How to use:

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.diag([10, 1])
    R = np.diag([[0.1]]).reshape(1, 1)
    T = 10
    dt = 0.01
    lqr = LQR(A, B, Q, R, T=T, dt=dt, mode="time-varying")
    time, x, u, x_d_all, K_hist = simulate_lqr(lqr, T, dt)


Code: Reza Azadeh - 2025
