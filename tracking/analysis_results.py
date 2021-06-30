import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []


# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name='lasot',
#                             run_ids=None, display_name='STARK-ST101'))

# trackers.extend(trackerlist(name='stark_p', parameter_name='baseline', dataset_name='otb',
#                             run_ids=5, display_name='STARK-P'))
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name='otb',
#                             run_ids=6, display_name='STARK-S50'))

trackers.extend(trackerlist(name='stark_p', parameter_name='baseline', dataset_name='lasot',
                            run_ids=1, display_name='STARK-P'))
trackers.extend(trackerlist(name='stark_p_local', parameter_name='baseline', dataset_name='lasot',
                            run_ids=4, display_name='STARK-P-LOCAL'))
trackers.extend(trackerlist(name='stark_p_no_decoder_t', parameter_name='baseline', dataset_name='lasot',
                            run_ids=8, display_name='STARK-P-NoDecoderT'))
trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name='lasot',
                            run_ids=None, display_name='STARK-S50'))
trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name='lasot',
                            run_ids=1, display_name='STARK-ST50'))





dataset = get_dataset('lasot')
plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'norm_prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
# print_per_sequence_results(trackers, dataset, report_name="debug")
