import numpy as np
from math import cos, sin, atan2, sqrt, pi, asin
from scipy.integrate import odeint
from verse.agents import BaseAgent
from verse.parser.parser import ControllerIR
import copy
import matplotlib.pyplot as plt 

class FixedWingAgent3(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # super().__init__(id, code, file_name)
        self.id = id
        self.init_cont = None
        self.init_disc = None
        self.static_parameters = None 
        self.uncertain_parameters = None
        self.decision_logic = ControllerIR.empty()
        self.K1 = [0.01,0.001,0.01,0.01]
        self.K2 = [0.005,0.005]
        self.scenarioType = '2D'
        # self.safeTraj = ctrlArgs[2]
        self.cst_input = [pi/18,0,0]
        # self.predictedSimulation = None
        self.estimated_state = None

    def aircraft_dynamics(self, state, t):
        # This function are the "tracking" dynamics used for the dubin's aircraft
        x,y,z,heading, pitch, velocity = state
        x_est,y_est,z_est,heading_est, pitch_est, velocity_est = self.estimated_state
        
        headingInput, pitchInput, accelInput = self.cst_input

        heading = heading%(2*pi)
        if heading > pi:
            heading = heading - 2*pi
        pitch = pitch%(2*pi)
        if pitch > pi:
            pitch = pitch - 2*pi

        heading_est = heading_est%(2*pi)
        if heading_est > pi:
            heading_est = heading_est - 2*pi
        pitch_est = pitch_est%(2*pi)
        if pitch_est > pi:
            pitch_est = pitch_est - 2*pi


        xref, yref, zref, headingref, pitchref, velref = self.goal_state
        # print(f"Goal state: {xref}; Estimate state: {x}")
        x_err = cos(heading_est)*(xref - x_est) + sin(heading_est)*(yref - y_est)
        y_err = -sin(heading_est)*(xref - x_est) + cos(heading_est)*(yref - y_est)
        z_err = zref - z_est
        heading_err = headingref - heading_est

        new_vel_xy = velref*cos(pitchref)*cos(heading_err)+self.K1[0]*x_err
        new_heading_input = heading_err + velref*(self.K1[1]*y_err + self.K1[2]*sin(heading_err))
        new_vel_z = velref*sin(pitchref)+self.K1[3]*z_err
        new_vel = sqrt(new_vel_xy**2 + new_vel_z**2)

        headingInput = new_heading_input
        accelInput = self.K2[0]*(new_vel - velocity)
        pitchInput = (pitchref - pitch_est) + (self.K2[1]*z_err)

        # if 'SAFETY' in str(mode[0]):
        #     if velocity <= 70:
        #         accelInput = 0
        #     else:
        #         accelInput = -10

        # Time derivative of the states
        dxdt = velocity*cos(heading)*cos(pitch)
        dydt = velocity*sin(heading)*cos(pitch)
        dzdt = velocity*sin(pitch)
        dheadingdt = headingInput
        dpitchdt = pitchInput
        dveldt = accelInput

        # print(dxdt, dydt, dzdt)
        # if dxdt < 0:
        #     print("stop")

        accel_max = 10
        heading_rate_max = pi/18
        pitch_rate_max = pi/18

        if abs(dveldt)>accel_max:
            dveldt = np.sign(dveldt)*accel_max
        if abs(dpitchdt)>pitch_rate_max*1:
            dpitchdt = np.sign(dpitchdt)*pitch_rate_max
        if abs(dheadingdt)>heading_rate_max:
            dheadingdt = np.sign(dheadingdt)*heading_rate_max

        return [dxdt, dydt, dzdt, dheadingdt, dpitchdt, dveldt]

        
    def simulate(self, initial_state, time_step, time_horizon):
        sol = odeint(self.aircraft_dynamics, initial_state, np.linspace(0, time_step, 2))
        # print(sol)
        # print("Solved Trajectory: ", sol)
        return sol

    def step(self, cur_state_estimated, initial_condition, time_step, goal_state):
        # if 'UNTRUSTED' in str(mode[0]):
        #     self.goal_state = self.desiredTraj(simulatorState)
        # else:
        #     self.goal_state = self.safeTraj(simulatorState)
        self.goal_state = goal_state
        self.estimated_state = cur_state_estimated
        initial_condition[3] = initial_condition[3]%(2*pi)
        sol = self.simulate(initial_condition, time_step, time_step)
        # print("")
        return list(sol[-1])

    # def TC_simulate(self, mode, initial_condition, time_horizon, time_step, lane_map=None):
    #     # TC simulate function for getting reachable sets
    #     trace = []

    #     new_states = self.simulate(initial_condition, time_step, time_horizon)
        
    #     for i, new_state in enumerate(new_states):
    #         trace.append([i * time_step] + list(new_state))

    #     return np.array(trace)

    def run_ref(self, ref_state, time_step, approaching_angle=3):
        k = np.tan(approaching_angle*(np.pi/180))
        delta_x = ref_state[-1]*time_step
        delta_z = k*delta_x
        return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])

    def TC_simulate(self, mode, initial_condition, time_horizon, time_step, lane_map=None):
        time_steps = np.arange(0,time_horizon+time_step/2, time_step)

        state = np.array(initial_condition)
        trajectory = copy.deepcopy(state)
        trajectory = np.insert(trajectory, 0, time_steps[0])
        trajectory = np.reshape(trajectory, (1,-1))
        for i in range(1, len(time_steps)):
            x_ground_truth = state[:6]
            x_estimate = state[6:12]
            ref_state = state[12:]
            x_next = self.step(x_estimate, x_ground_truth, time_step, ref_state)
            x_next[3] = x_next[3]%(np.pi*2)
            if x_next[3] > np.pi:
                x_next[3] = x_next[3]-np.pi*2
            ref_next = self.run_ref(ref_state, time_step, approaching_angle=3)
            # print(ref_next)
            state = np.concatenate((x_next, x_estimate, ref_next))
            tmp = np.insert(state, 0, time_steps[i])
            tmp = np.reshape(tmp, (1,-1))
            trajectory = np.vstack((trajectory, tmp))

        return trajectory

if __name__ == "__main__":
    agent = FixedWingAgent3('a')
    init = np.array([-3000, 10, 120.0, 0, -np.deg2rad(3), 0, -3000, 10, 120.0, 0, -np.deg2rad(3), 0, -3000, 0, 120, 0, -np.deg2rad(3), 10])
    traj = agent.TC_simulate(None, init, 15, 0.05, None)
    print(traj)

    plt.figure()
    plt.plot(traj[:,1], traj[:,2])

    plt.figure()
    plt.plot(traj[:,1], traj[:,3])

    plt.show()
# import numpy as np
# from math import cos, sin, atan2, sqrt, pi, asin
# from scipy.integrate import odeint
# from verse.agents import BaseAgent
# from verse.parser.parser import ControllerIR
# import copy
# import matplotlib.pyplot as plt 

# class FixedWingAgent3(BaseAgent):
#     def __init__(self, id, code=None, file_name=None):
#         # super().__init__(id, code, file_name)
#         self.id = id
#         self.init_cont = None
#         self.init_disc = None
#         self.static_parameters = None 
#         self.uncertain_parameters = None
#         self.decision_logic = ControllerIR.empty()
#         self.K1 = [0.01,0.01,0.01,0.01]
#         self.K2 = [0.005,0.005]
#         self.scenarioType = '2D'
#         # self.safeTraj = ctrlArgs[2]
#         self.cst_input = [pi/18,0,0]
#         # self.predictedSimulation = None
#         self.estimated_state = None

#     def aircraft_dynamics(self, state, t):
#         # This function are the "tracking" dynamics used for the dubin's aircraft
#         x,y,z,heading, pitch, velocity = state
#         x_est, y_est, z_est, heading_est, pitch_est, velocity_est = self.estimated_state
#         headingInput, pitchInput, accelInput = self.cst_input

#         heading = heading%(2*pi)
#         if heading > pi:
#             heading = heading - 2*pi
#         pitch = pitch%(2*pi)
#         if pitch > pi:
#             pitch = pitch - 2*pi


#         xref, yref, zref, headingref, pitchref, velref = self.goal_state
#         # print(f"Goal state: {xref}; Estimate state: {x}")
#         x_err = cos(heading_est)*(xref - x_est) + sin(heading_est)*(yref - y_est)
#         y_err = -sin(heading_est)*(xref - x_est) + cos(heading_est)*(yref - y_est)
#         z_err = zref - z_est
#         heading_err = headingref - heading_est

#         new_vel_xy = velref*cos(pitchref)*cos(heading_err)+self.K1[0]*x_err
#         new_heading_input = heading_err + velref*(self.K1[1]*y_err + self.K1[2]*sin(heading_err))
#         new_vel_z = velref*sin(pitchref)+self.K1[3]*z_err
#         new_vel = sqrt(new_vel_xy**2 + new_vel_z**2)

#         headingInput = new_heading_input
#         accelInput = self.K2[0]*(new_vel - velocity_est)
#         pitchInput = (pitchref - pitch_est) + (self.K2[1]*z_err)

#         # if 'SAFETY' in str(mode[0]):
#         #     if velocity <= 70:
#         #         accelInput = 0
#         #     else:
#         #         accelInput = -10

#         # Time derivative of the states
#         dxdt = velocity*cos(heading)*cos(pitch)
#         dydt = velocity*sin(heading)*cos(pitch)
#         dzdt = velocity*sin(pitch)
#         dheadingdt = headingInput
#         dpitchdt = pitchInput
#         dveldt = accelInput

#         # print(dxdt, dydt, dzdt)
#         # if dxdt < 0:
#         #     print("stop")

#         accel_max = 10
#         heading_rate_max = pi/18
#         pitch_rate_max = pi/18

#         if abs(dveldt)>accel_max:
#             dveldt = np.sign(dveldt)*accel_max
#         if abs(dpitchdt)>pitch_rate_max*1:
#             dpitchdt = np.sign(dpitchdt)*pitch_rate_max
#         if abs(dheadingdt)>heading_rate_max:
#             dheadingdt = np.sign(dheadingdt)*heading_rate_max

#         return [dxdt, dydt, dzdt, dheadingdt, dpitchdt, dveldt]

        
#     def simulate(self, initial_state, time_step, time_horizon):
#         sol = odeint(self.aircraft_dynamics, initial_state, np.linspace(0, time_step, 2))
#         # print(sol)
#         # print("Solved Trajectory: ", sol)
#         return sol

#     def step(self, cur_state_estimated, initial_condition, time_step, goal_state):
#         # if 'UNTRUSTED' in str(mode[0]):
#         #     self.goal_state = self.desiredTraj(simulatorState)
#         # else:
#         #     self.goal_state = self.safeTraj(simulatorState)
#         self.goal_state = goal_state
#         self.estimated_state = cur_state_estimated
#         initial_condition[3] = initial_condition[3]%(2*pi)
#         sol = self.simulate(initial_condition, time_step, time_step)
#         # print("")
#         return list(sol[-1])

#     # def TC_simulate(self, mode, initial_condition, time_horizon, time_step, lane_map=None):
#     #     # TC simulate function for getting reachable sets
#     #     trace = []

#     #     new_states = self.simulate(initial_condition, time_step, time_horizon)
        
#     #     for i, new_state in enumerate(new_states):
#     #         trace.append([i * time_step] + list(new_state))

#     #     return np.array(trace)

#     def run_ref(self, ref_state, time_step, approaching_angle=3):
#         k = np.tan(approaching_angle*(np.pi/180))
#         delta_x = ref_state[-1]*time_step
#         delta_z = k*delta_x # *time_step
#         return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])

#     def TC_simulate(self, mode, initial_condition, time_horizon, time_step, lane_map=None):
#         time_steps = np.arange(0,time_horizon+time_step/2, time_step)

#         state = np.array(initial_condition)
#         trajectory = copy.deepcopy(state)
#         trajectory = np.insert(trajectory, 0, time_steps[0])
#         trajectory = np.reshape(trajectory, (1,-1))
#         for i in range(1, len(time_steps)):
#             x_ground_truth = state[:6]
#             x_estimate = state[6:12]
#             ref_state = state[12:]
#             x_next = self.step(x_estimate, x_ground_truth, time_step, ref_state)
#             x_next[3] = x_next[3]%(np.pi*2)
#             if x_next[3] > np.pi:
#                 x_next[3] = x_next[3]-np.pi*2
#             ref_next = self.run_ref(ref_state, time_step, approaching_angle=3)
#             # print(ref_next)
#             state = np.concatenate((x_next, x_estimate, ref_next))
#             tmp = np.insert(state, 0, time_steps[i])
#             tmp = np.reshape(tmp, (1,-1))
#             trajectory = np.vstack((trajectory, tmp))

#         return trajectory

# if __name__ == "__main__":
#     agent = FixedWingAgent3('a')
#     init = np.array([-2550, 10, 120.0, 0, -np.deg2rad(3), 0, -2500, 0, 120, 0, -np.deg2rad(3), 50])
#     traj = agent.TC_simulate(None, init, 15, 0.05, None)
#     print(traj)

#     plt.figure()
#     plt.plot(traj[:,1], traj[:,2])

#     plt.figure()
#     plt.plot(traj[:,1], traj[:,3])

#     plt.show()