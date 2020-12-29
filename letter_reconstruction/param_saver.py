from datetime import datetime

import utils
from data.data_dir import data_dir_path


def save_params(model):
    print('Saving...')
    params = {
        'bias': model.conv.bias.detach().numpy() if model.conv.bias is not None else None,
        'weight': model.conv.weight.detach().numpy(),
    }
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'{data_dir_path}/w_{date_str}.pickle'
    utils.save_pickle(params, path)
    print('Done')
