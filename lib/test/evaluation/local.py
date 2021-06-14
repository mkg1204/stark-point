from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/zikun/data/mkg/Projects/Stark/data/got10k_lmdb'
    settings.got10k_path = '/home/zikun/data/mkg/Projects/Stark/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/zikun/data/mkg/Projects/Stark/data/lasot_lmdb'
    settings.lasot_path = '/home/zikun/data/mkg/Projects/Stark/data/lasot'
    settings.network_path = '/home/zikun/data/mkg/Projects/Stark/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/home/zikun/data/mkg/Projects/Stark'
    settings.result_plot_path = '/home/zikun/data/mkg/Projects/Stark/test/result_plots'
    settings.results_path = '/home/zikun/data/mkg/Projects/Stark/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/zikun/data/mkg/Projects/Stark'
    settings.segmentation_path = '/home/zikun/data/mkg/Projects/Stark/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/zikun/data/mkg/Projects/Stark/data/trackingNet'
    settings.uav_path = ''
    settings.vot_path = '/home/zikun/data/mkg/Projects/Stark/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

