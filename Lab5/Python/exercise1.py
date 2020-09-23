""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import farms_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

plt.close("all")

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")

    # Create muscle object
    muscle = Muscle(parameters)

    pylog.warning("Isometric muscle contraction to be completed")

    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.l_opt # To get the muscle optimal length

    # Evalute for a single muscle stretch
    muscle_stretch = 0.35

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    sys.muscle.l_opt=0.15
    # Set the initial condition
    x0 = [0.0, sys.muscle.l_opt]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value
    print(x0)

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           stimulation=muscle_stimulation,
                           muscle_length=muscle_stretch)
   
    # Plotting
    plt.figure('Isometric muscle experiment')
    plt.plot(result.l_ce/sys.muscle.l_opt, result.active_force, label = "Active Force")
    plt.plot(result.l_ce/sys.muscle.l_opt, result.passive_force, label = "Passive Force")
    plt.plot(result.l_ce/sys.muscle.l_opt, result.passive_force + result.active_force, label = "Total Force")
    plt.legend(loc="upper left")
    plt.title('Isometric muscle experiment')
    plt.xlabel('length')
    plt.ylabel('Force [N]')
    plt.grid()
    plt.show
    
    
    plt.figure('Isometric Muscle experiment 2')
    
    stim = np.linspace(0,1,11)
    print(stim)
    for i in range(10):
        muscle_stimulation = stim[i]
        j = 0.1*i
        j = round(j,1)
        # Run the integration
        result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           stimulation=muscle_stimulation,
                           muscle_length=muscle_stretch)
    
    
        plt.plot(result.l_ce,result.active_force, label ="MS %s" %j)
    a
    plt.xlabel('length')
    plt.ylabel('Force [N]')
    plt.show
    
    
def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.l_opt # To get the muscle optimal length

    # Evalute for a single load
    load = 250/9.81

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.0

    # Set the initial condition
    x0 = [0.0, sys.muscle.l_opt,
          sys.muscle.l_opt + sys.muscle.l_slack, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.4
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)
    
    load = np.linspace(100/9.81,1700/9.81,50)
    Stimu = np.linspace(0,1,6)
    Vce = np.zeros((len(load),len(Stimu)))
    plt.figure('Isotonix Globale')
    for j in range(len(Stimu)):
        
   #     Vce = np.empty(len(load))
    #    Vcemax = np.empty(len(load))
    #    Vcemin = np.empty(len(load))
    
        for i in range(len(load)):

    # Run the integration
            result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           time_stabilize=time_stabilize,
                           stimulation=Stimu[j],
                           load=load[i]
                           )
            print(result.l_mtu[-1])
            print(sys.muscle.l_opt + sys.muscle.l_slack)
            if (result.l_mtu[-1] > sys.muscle.l_opt + sys.muscle.l_slack):
                Vce[i,j]=(max(result.v_ce))
            else: Vce[i,j] = min(result.v_ce)
        
        Vce[:,j]
        plt.plot(Vce[:,j],load, label ="MS %s" %round(Stimu[j],1))
        plt.legend(loc = "upper left")
        plt.xlabel('Vitesse max [m/s]')
        plt.ylabel('Load [kg]')






    # Plotting
    plt.figure('Isotonic muscle experiment')
    #plt.plot(result.time,
    #         result.v_ce)
    plt.plot(Vce,load)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Vitesse max [m/s]')
    plt.ylabel('Load [kg]')
    #plt.xlabel('Time [s]')
    #plt.ylabel('Muscle contracticle velocity [lopts/s]')
    plt.grid()

#MUSCLE-Force Velocity relationship : 
    plt.figure('Muscle Force-Velocity')
    plt.plot(result.v_ce,result.active_force, label ="Active Force")
    plt.plot(result.v_ce,result.active_force+result.passive_force, label = "Passive + Active")
    plt.plot(result.v_ce,result.passive_force, label = "Passive Force")
    plt.legend(loc = "upper left")
    plt.xlabel('Vitesse m')
    plt.ylabel('Force')
    

def exercise1():
    exercise1a()
    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

