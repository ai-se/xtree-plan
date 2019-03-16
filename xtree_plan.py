from src.experiments.experiment1 import Experiment1
from src.data.get_data import get_all_projects
import multiprocessing as mp
from pdb import set_trace

if __name__ == "__main__":
    data = get_all_projects()

    # -- Initialize the experiment --
    exp = Experiment1(verbose=True, plot_results=False, decrease=False)

    # # -- Serial Execution --
    # for proj, path in data.items():
    #     exp.main(proj, path)

    # -- Get CPU core count --
    n_proc = mp.cpu_count()

    # -- Run asynchronously --
    with mp.Pool(processes=n_proc) as pool:
        tasks = pool.starmap(exp.main, tuple(data.items()))
