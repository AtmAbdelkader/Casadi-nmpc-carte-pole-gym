import numpy as np
import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit

env = gym.make('CartPole-v1', render_mode='human', disable_env_checker=True)
env = TimeLimit(env.env, max_episode_steps=1000)

g = 9.81
m_c = 1.0
m_p = 0.1
l = 0.5
dt = 0.02

n_states = 4
n_controls = 1

x = ca.MX.sym('x', n_states)
u = ca.MX.sym('u', n_controls)

sintheta = ca.sin(x[2])
costheta = ca.cos(x[2])
total_mass = m_c + m_p
temp = (u[0] + m_p * l * x[3]**2 * sintheta) / total_mass
theta_acc = (g * sintheta - costheta * temp) / (l * (4.0/3.0 - m_p * costheta**2 / total_mass))
x_acc = temp - m_p * l * theta_acc * costheta / total_mass
xdot = ca.vertcat(x[1], x_acc, x[3], theta_acc)

k1 = xdot
k2 = ca.substitute(xdot, x, x + dt/2 * k1)
k3 = ca.substitute(xdot, x, x + dt/2 * k2)
k4 = ca.substitute(xdot, x, x + dt * k3)
x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
f = ca.Function('f', [x, u], [x_next])

N = 30
opti = ca.Opti()
X = opti.variable(n_states, N+1)
U = opti.variable(n_controls, N)

x0_param = opti.parameter(n_states)
x_ref_param = opti.parameter(n_states)

Q = ca.diag([100, 0.01, 100, 0.01])
R = ca.diag([0.0001])
obj = 0
for k in range(N):
    obj += ca.mtimes([(X[:,k]-x_ref_param).T, Q, (X[:,k]-x_ref_param)])
    obj += ca.mtimes([U[:,k].T, R, U[:,k]])
    opti.subject_to(X[:,k+1] == f(X[:,k], U[:,k]))

opti.minimize(obj)
opti.subject_to(X[:,0] == x0_param)

u_max = 20.0
opti.subject_to(opti.bounded(-u_max, U, u_max))
opts = {'ipopt.print_level':0, 'print_time':0}
opti.solver('ipopt', opts)

# Simulation
obs, _ = env.reset(seed=42)
x_current = np.array(obs)
x_target = np.array([0, 0, 0, 0])

x_log = [x_current]
u_log = []

for i in range(10000):
    opti.set_value(x0_param, x_current)
    opti.set_value(x_ref_param, x_target)

    try:
        sol = opti.solve()
        u_opt = sol.value(U[:,0])
    except:
        u_opt = np.array([0])

    u_apply = np.clip(u_opt, -u_max, u_max)
    u_log.append(u_apply.item())

    action = int(u_apply.item() > 0)
    obs, reward, done, truncated, info = env.step(action)
    x_current = np.array(obs)
    x_log.append(x_current)

    if done or truncated:
        break

env.close()

# ----------------------------------------
# 📈 Graphs of results :
# ----------------------------------------
x_log = np.array(x_log)
u_log = np.array(u_log)
t = np.linspace(0, dt * len(u_log), len(u_log))

fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
axs[0].plot(t, x_log[:-1, 0], label='Position')
axs[0].set_ylabel("x (m)")
axs[0].grid()

axs[1].plot(t, x_log[:-1, 1], label='Velocity')
axs[1].set_ylabel("ẋ (m/s)")
axs[1].grid()

axs[2].plot(t, x_log[:-1, 2], label='Angle')
axs[2].set_ylabel("θ (rad)")
axs[2].grid()

axs[3].plot(t, x_log[:-1, 3], label='Angular velocity')
axs[3].set_ylabel("θ̇ (rad/s)")
axs[3].grid()

axs[4].plot(t, u_log, label='Control')
axs[4].set_ylabel("u (N)")
axs[4].set_xlabel("Time (s)")
axs[4].grid()

plt.tight_layout()
plt.suptitle("MPC Control of CartPole", fontsize=16, y=1.02)
plt.show()
