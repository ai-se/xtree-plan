from src.experiments.experiment1 import Experiment1

if __name__ == "__main__":
    exp = Experiment1(verbose=True, plot_results=True, decrease=False)
    exp.main()
