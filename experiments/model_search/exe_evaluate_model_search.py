from DLBio.helpers import MyDataFrame, search_rgx, load_json, check_mkdir
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from DLBio.kwargs_translator import get_kwargs

BASE_FOLDER = 'experiments/model_search'  # the folder of the experiment
MODEL_FOLDER = join(BASE_FOLDER, 'trained_models')
IMAGE_FOLDER = join(BASE_FOLDER, 'boxplots')

# regular expression to find all folders that contain trained models
RGX = r'\d\d\d\d'

# Create groups that are separated by these keywords. Aggregation over all
# seeds.
GROUP_KEYS = ['lr', 'num_layers', 'kernel_size', 'hidden_dim']
# aggregate these keywords: compute the mean and standard deviation
AGG = {
    'last_train_error': ('mean', 'std'), 'last_val_error': ('mean', 'std'),
    'min_val_error': ('mean', 'std')
}


def run():
    # find all folders matching the regular expression in model folder
    folders_ = search_rgx(RGX, MODEL_FOLDER)
    assert folders_

    # create a dataframe: a table with all results
    df = MyDataFrame()
    for folder in folders_:
        folder = join(MODEL_FOLDER, folder)
        df = update(df, folder)

    # convert to a pandas Dataframe
    df = df.get_df()
    # aggregate and sort by specific keys
    df = df.groupby(GROUP_KEYS, as_index=False).agg(AGG)
    df = df.sort_values(
        [('last_val_error', 'mean'), ('min_val_error', 'mean')]
    )
    create_boxplots(df)

    # save as comma-separated file
    df.to_csv(join(BASE_FOLDER, 'results.csv'))

    # write the ten best configurations as a markdown table that you can
    # copy and paste into the README.md file directly
    with open(join(BASE_FOLDER, 'table.md'), 'w') as file:
        file.write(df.head(10).to_markdown())


def create_boxplots(df):
    # create boxplots for different keys
    for y_key in ['last_train_error', 'last_val_error', 'min_val_error']:
        y_key = tuple([y_key, 'mean'])
        for x_key in ['lr', 'num_layers', 'kernel_size', 'hidden_dim']:
            plt.figure()

            sns.boxplot(data=df, x=x_key, y=y_key)
            plt.grid()
            plt.tight_layout()

            out_path = join(IMAGE_FOLDER, f'bp_{x_key}_{y_key}.png')
            check_mkdir(out_path)
            plt.savefig(out_path)
            plt.close()


def update(df, folder):
    log = load_json(join(folder, 'log.json'))
    if log is None:
        return df

    opt = load_json(join(folder, 'opt.json'))
    if opt is None:
        return df

    model_kw = get_kwargs(opt['model_kw'])

    df.update({
        'lr': opt['lr'],
        'num_layers': model_kw['num_layers'][0],
        'kernel_size': model_kw['kernel_size'][0],
        'hidden_dim': model_kw['hidden_dim'][0],
        'last_train_error': log['er'][-1],
        'last_val_error': log['val_er'][-1],
        'min_val_error': min(log['val_er']),
        'seed': opt['seed']
    })

    return df


if __name__ == "__main__":
    run()
