import tomllib
import pygame
import numpy as np
from matplotlib import pyplot as plt

with open("config.toml", mode="rb") as fp:
    toml_data = tomllib.load(fp)

COLLISIONS = toml_data["settings"]["collisions"]
SCREEN_WIDTH = toml_data["graphics"]["screen_width"]
SCREEN_HEIGHT = SCREEN_WIDTH * 3/4

g = toml_data["fixed_params"]["gravity"] # Acceleration due to gravity
m1 = toml_data["fixed_params"]["mass1"] # Mass of the 1st object
m2 = toml_data["fixed_params"]["mass2"] # Mass of the 2nd object

l1 = toml_data["fixed_params"]["length1"] # Length of the string
                                          # connecting the roof and the 1st object

l2 = toml_data["fixed_params"]["length2"] # Length of the string 
                                          # connecting the two objects

l1_adjusted = SCREEN_HEIGHT * 0.8 * l1/(l1 + l2)
l2_adjusted = SCREEN_HEIGHT * 0.8 * l2/(l1 + l2)

m1_relative =  m1 / (m1 + m2)
m2_relative =  m2 / (m1 + m2)
proportional_size = np.sqrt((SCREEN_WIDTH * SCREEN_HEIGHT * 5.1e-4))*3

m1_size = np.max([m1_relative * proportional_size, m2_relative * proportional_size*0.33]) 
m2_size = np.max([m2_relative * proportional_size, m1_relative * proportional_size*0.33])

epsilon = np.min([m1_size, m2_size])/10

mass1_dim = (m1_size, m1_size)
mass2_dim = (m2_size, m2_size)
roof_center = (SCREEN_WIDTH / 2, SCREEN_HEIGHT * 0.1)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Double Pendulum Simulation")
clock = pygame.time.Clock()

roof = pygame.surface.Surface((SCREEN_WIDTH * 0.97, SCREEN_HEIGHT * 0.05))
if COLLISIONS is False:
    roof = pygame.surface.Surface((SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.05))
mass1 = pygame.surface.Surface(mass1_dim)
mass2 = pygame.surface.Surface(mass2_dim)

roof.fill("#F9F6EE")
mass1.fill("#F9F6EE")
mass2.fill("#F9F6EE")

varphi1 = toml_data["initial"]["varphi1"] # Angle from the vertical axis
varphi2 = toml_data["initial"]["varphi2"]

varphi1_dot = toml_data["initial"]["varphi1_dot"] # Angular velocity
varphi2_dot = toml_data["initial"]["varphi2_dot"]

varphi1_ddot = toml_data["initial"]["varphi1_ddot"] # Angular acceleration
varphi2_ddot = toml_data["initial"]["varphi2_ddot"]

# For testing purposes - potential and kinetic energy
total_energy = -(m1 + m2) * g * l1 * np.cos(varphi1) - m2 * g * l2 * np.cos(varphi2) \
                    + (0.5 * (m1 + m2) * (l1**2) * (varphi1_dot**2) 
                    + 0.5 * m2 * (l2**2) * (varphi1_dot**2) \
                    + m2 * l1 * l2 * np.cos(varphi2 - varphi1) * varphi1_dot * varphi2_dot)
kinetic_data = []
potential_data = []
time = []

dt = 0.005 # A small time step

mass1pos_trace = [] # For using the trace effect
mass2pos_trace = []

# For detecting collisions, inside status
mass1_inside_up = False
mass1_inside_lateral = False
mass2_inside_up = False
mass2_inside_lateral = False

iteration = 0
RUN = True

def acceleration(x1, x2, x1_dot, x2_dot) -> float:
    """Function for calculating acceleration of a point mass"""
    C1 = m2_relative * l2/l1 * np.cos(x2 - x1)
    C2 = g/l1 * np.sin(x1) - m2_relative * l2/l1 * np.sin(x2 - x1) * x2_dot**2 
    K1 = l1/l2 * np.cos(x2 - x1)
    K2 = l1/l2 * np.sin(x2 - x1) * x1_dot**2 + g/l2 * np.sin(x2)
    return ((C1 * K2 - C2) / (1 - C1 * K1), (K1 * C2 - K2) / (1 - C1 * K1))

while RUN:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            RUN = False

    if toml_data["settings"]["integration"] == "euler":
        varphi1_ddot, varphi2_ddot = acceleration(varphi1, varphi2, varphi1_dot, varphi2_dot)

        varphi1 += varphi1_dot * dt + 0.5 * varphi1_ddot * dt**2
        varphi2 += varphi2_dot * dt + 0.5 * varphi2_ddot * dt**2

        varphi1_dot += 0.5 * varphi1_ddot * dt
        varphi2_dot += 0.5 * varphi2_ddot * dt

    elif toml_data["settings"]["integration"] == "leapfrog":
        varphi1_ddot_new, varphi2_ddot_new = acceleration(varphi1, varphi2, varphi1_dot, varphi2_dot)

        varphi1 += varphi1_dot * dt + 0.5 * varphi1_ddot * dt**2
        varphi2 += varphi2_dot * dt + 0.5 * varphi2_ddot * dt**2

        varphi1_dot += 0.5 * (varphi1_ddot + varphi1_ddot_new) * dt
        varphi2_dot += 0.5 * (varphi2_ddot + varphi2_ddot_new) * dt

        varphi1_ddot = varphi1_ddot_new
        varphi2_ddot = varphi2_ddot_new

    elif toml_data["settings"]["integration"] == "runge-kutta":
        varphi1_ddot, varphi2_ddot = acceleration(varphi1, varphi2, varphi1_dot, varphi2_dot)

        # runge-kutta 4th order coefficients for the first object
        k1v = varphi1_ddot * dt
        k1x = varphi1_dot * dt
        k2v = acceleration(varphi1 + 0.5 * k1x, varphi2, varphi1_dot + 0.5 * k1v, varphi2_dot)[0] * dt
        k2x = (varphi1_dot + 0.5 * k1v) * dt
        k3v = acceleration(varphi1 + 0.5 * k2x, varphi2, varphi1_dot + 0.5 * k2v, varphi2_dot)[0] * dt
        k3x = (varphi1_dot + 0.5 * k2v) * dt
        k4v = acceleration(varphi1 + k3x, varphi2, varphi1_dot + k3v, varphi2_dot)[0] * dt
        k4x = (varphi1_dot + k3v) * dt

        varphi1_dot += 1/6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        varphi1 += 1/6 * (k1x + 2 * k2x + 2 * k3x + k4x)

        # runge-kutta 4th order coefficients for the second object
        k1v = varphi2_ddot * dt
        k1x = varphi2_dot * dt
        k2v = acceleration(varphi1, varphi2 + 0.5 * k1x, varphi1_dot, varphi2_dot + 0.5 * k1v)[1] * dt
        k2x = (varphi2_dot + 0.5 * k1v) * dt
        k3v = acceleration(varphi1, varphi2 + 0.5 * k2x, varphi1_dot, varphi2_dot + 0.5 * k2v)[1] * dt
        k3x = (varphi2_dot + 0.5 * k2v) * dt
        k4v = acceleration(varphi1, varphi2 + k3x, varphi1_dot, varphi2_dot + k3v)[1] * dt
        k4x = (varphi2_dot + k3v) * dt

        varphi2_dot += 1/6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        varphi2 += 1/6 * (k1x + 2 * k2x + 2 * k3x + k4x)

    mass1_pos = (roof_center[0] + l1_adjusted * np.sin(varphi1),
            roof_center[1] + l1_adjusted * np.cos(varphi1) + mass1_dim[0]/2)
    mass2_pos = (roof_center[0] + l1_adjusted * np.sin(varphi1) + l2_adjusted * np.sin(varphi2),
            roof_center[1] + l1_adjusted * np.cos(varphi1) + l2_adjusted * np.cos(varphi2) + mass1_dim[0] + mass2_dim[0]/2)
    

    if COLLISIONS:
        # Colllisions in the x axis
        if mass1_pos[0] + mass1_dim[0] >= SCREEN_WIDTH - epsilon and mass1_inside_lateral == False:
            varphi1_dot = (-1) * varphi1_dot
            varphi1_ddot = 0
            mass1_inside_lateral = True
            print("AAAA")

        elif mass1_pos[0] - mass1_dim[0] <= epsilon and mass1_inside_lateral == False:
            varphi1_dot = (-1) * varphi1_dot
            varphi1_ddot = 0
            mass1_inside_lateral = True
            print("AAAA")
    
        elif mass1_pos[0] - mass1_dim[0] > 0 and mass1_pos[0] + mass1_dim[0] < SCREEN_WIDTH:
            mass1_inside_lateral = False
        
        if mass2_pos[0] + mass2_dim[0] >= SCREEN_WIDTH - epsilon and mass2_inside_lateral == False:
            varphi2_dot = (-1) * varphi2_dot
            varphi2_ddot = 0
            mass2_inside_lateral = True
            print("AAAA")

        elif mass2_pos[0] - mass2_dim[0] <= epsilon and mass2_inside_lateral == False:
            varphi2_dot = (-1) * varphi2_dot
            varphi2_ddot = 0
            mass2_inside_lateral = True
            print("AAAA")
    
        elif mass2_pos[0] - mass2_dim[0] > 0 and mass2_pos[0] + mass2_dim[0] < SCREEN_WIDTH:
            mass2_inside_lateral = False
            
        # Colllisions in the y axis
        if mass1_pos[1] - mass1_dim[1] <= roof_center[1] and mass1_inside_up == False:
            varphi1_dot = (-1) * varphi1_dot
            varphi1_ddot = 0
            mass1_inside_up = True
            print("AAAA")

        elif mass1_pos[1] - mass1_dim[1] > roof_center[1]:
            mass1_inside_up = False

        if mass2_pos[1] - mass2_dim[1] <= roof_center[1] and mass2_inside_up == False:
            varphi2_dot = (-1) * varphi2_dot
            varphi2_ddot = 0
            mass2_inside_up = True
            print("AAAA")

        elif mass2_pos[1] - mass2_dim[1] > roof_center[1]:
            mass2_inside_up = False

    pe = -(m1 + m2) * g * l1 * np.cos(varphi1) - m2 * g * l2 * np.cos(varphi2)
    ke = (0.5 * (m1 + m2) * (l1**2) * (varphi1_dot**2) 
                    + 0.5 * m2 * (l2**2) * (varphi1_dot**2)
                    + m2 * l1 * l2 * np.cos(varphi2 - varphi1) * varphi1_dot * varphi2_dot)

    kinetic_data.append(ke)
    potential_data.append(pe)
    time.append(dt * iteration)
    screen.fill("#222222")

    if COLLISIONS:
        screen.blit(roof, (SCREEN_WIDTH*0.015,SCREEN_HEIGHT*0.05))
    else:
        screen.blit(roof, (SCREEN_WIDTH*0.4,SCREEN_HEIGHT*0.05))
    
    pygame.draw.line(screen, color="#F9F6EE", start_pos=roof_center, end_pos=mass1_pos, width=1)
    pygame.draw.line(screen, color="#F9F6EE", start_pos=mass1_pos, end_pos=mass2_pos, width=1)

    screen.blit(mass1, (mass1_pos[0] - mass1_dim[0]/2, mass1_pos[1] - mass1_dim[1]/2))
    screen.blit(mass2, (mass2_pos[0] - mass2_dim[0]/2, mass2_pos[1] - mass2_dim[1]/2))

    if iteration < SCREEN_WIDTH/100:
        mass1pos_trace.append(mass1_pos)
        mass2pos_trace.append(mass2_pos)

    else:
        mass1pos_trace.pop(0)
        mass2pos_trace.pop(0)
        mass1pos_trace.append(mass1_pos)
        mass2pos_trace.append(mass2_pos)

    for i in range(len(mass1pos_trace)-1):
        if i % 2 == 0:
            pygame.draw.line(screen, color="#F9F6EE", start_pos=mass1pos_trace[i], end_pos=mass1pos_trace[i+1])
            pygame.draw.line(screen, color="#F9F6EE", start_pos=mass2pos_trace[i], end_pos=mass2pos_trace[i+1])

    iteration += 1
    if bool(toml_data["settings"]["max_iter"]):
        if iteration > toml_data["settings"]["max_iter"]:
            RUN = False
    
    pygame.display.update()
    clock.tick(toml_data["graphics"]["fps"])
    
pygame.quit()

if toml_data["graphics"]["all_energy"]:
    plt.plot(time, kinetic_data)
    plt.plot(time, potential_data)
    plt.plot(time, np.array(kinetic_data) + np.array(potential_data))
    plt.legend(["kinetic energy", "potential energy", "total energy"])
else:
    plt.plot(time, np.array(kinetic_data) + np.array(potential_data))
    plt.plot(time, [total_energy for i in range(len(time))])
    plt.legend(["total energy", "initial energy"])

plt.title("Double pendulum energy")
plt.xlabel("time")
plt.ylabel("energy")

plt.show()
