from src.experiments.experiment1 import Experiment1
from src.data.get_data import get_all_projects
import multiprocessing as mp
from pdb import set_trace

if __name__ == "__main__":
    data = get_all_projects()
    projects = data.keys()
    # -- Get CPU core count --
    n_proc = mp.cpu_count()
    # -- Initialize the experiment --
    exp = Experiment1(verbose=True, plot_results=False, decrease=False)
    # -- Run asynchronously --
    # -- Create a pool object --
    with mp.Pool(processes=n_proc) as pool:
        tasks = pool.starmap(exp.main, tuple(data.items()))
