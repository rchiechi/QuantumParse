import pickle
import sys


def main():
    import matplotlib.pyplot as plt
    task, data = pickle.load(sys.stdin.buffer)
    if task == 'eos':
        from ase.eos import plot
        plot(*data)
    elif task == 'neb':
        forcefit = data
        forcefit.plot()
    elif task == 'reciprocal':
        from ase.dft.bz import bz_plot
        bz_plot(**data)
    elif task == 'graph':
        from ase.gui.graphs import make_plot
        make_plot(show=False, *data)
    else:
        print('Invalid task {}'.format(task))
        sys.exit(17)

    # Magic string to tell GUI that things went okay:
    print('GUI:OK')
    sys.stdout.close()

    plt.show()


if __name__ == '__main__':
    main()
