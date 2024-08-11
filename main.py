from Simulation import Simulation
from numerical_integration import ks_integration

X = ks_integration.int_plot(1000, periodicity_length=200,num_grid_points=512,time_step=0.25,plot=False,IC_seed=0)

single_sim = Simulation(X, 10**-8,500)


if __name__ == "__main__":
    single_sim.single_train()
    single_sim.predict_single(100)