import sys
import os
from OLE.utils.evaluation import plot_timings, plot_parameter


def main(argv):
    rootdir = argv[0]
    if rootdir == '-h':
        print('To use the example plotting script, provide your OLE output folder at the first argument, and optionally an MCMC parameter as second argument.')
        return 0
    elif rootdir == '--help':
        print('To use the example plotting script, provide your OLE output folder at the first argument, and optionally an MCMC parameter as second argument.')
        return 0
    logfile = os.path.join(rootdir, 'logfile_0.log')
    plot_dir = os.path.join(rootdir, 'OLE_plots/')
    print(f'Plotting information from logfile {logfile}, saving to {plot_dir}')
    plot_timings(logfile, plot_dir=plot_dir)
    if len(argv)>1:
        print(f'Plotting for parameter {argv[1]}, saving to {plot_dir}')
        plot_parameter([logfile], parameter=argv[1], plot_dir=plot_dir)

    



if __name__ == "__main__":
    if len(sys.argv)>1:
        main(sys.argv[1:])
    else:
        print('To use the example plotting script, provide your OLE output folder at the first argument, and optionally an MCMC parameter as second argument.')
   