# Casadi-nmpc-carte-pole-gym


---

🏗 CartPole Stabilization with Nonlinear MPC (CASADI)

This project uses CASADI to implement a Nonlinear Model Predictive Controller (NMPC) that balances the classic CartPole from gymnasium. The controller predicts future motion using the system’s nonlinear dynamics (integrated via RK4) and optimizes control actions over a moving horizon with the Ipopt solver. At each step, only the first optimal force is applied, keeping the pole upright. The code logs and plots 📊 the cart’s position, velocity, pole angle, angular velocity, and control force.


---

# 1️⃣ Install dependencies
pip install numpy casadi gymnasium matplotlib

# 2️⃣ Run the script
python3 cartpole_nmpc.py

___



![Screencast from 08-05-2025 02_19_31 PM](https://github.com/user-attachments/assets/68227684-dafd-4a65-903a-776b6962d71c)
