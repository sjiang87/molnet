from deephyper.datasets.gnn_datasets import molnet


def load_data(seed=0, full=False, test=False, **kwargs):
    """
    The function to load data.
    Args:
        seed: int, the random seed to separate dataset
        full: bool, if want full size data
        test: bool, if want test data
        **kwargs:
            data: str, the dateset name
            split: str, the split type
            ratio: int, fraction of the original data

    Returns:

    """
    data = kwargs['data']
    split = kwargs['split']
    ratio = int(kwargs['ratio'])
    print(f'Load dataset {data}, split {split}, seed {seed}, full {full}, test {test}.')

    [X_train, A_train, E_train, M_train, N_train, y_train], \
    [X_valid, A_valid, E_valid, M_valid, N_valid, y_valid], \
    [X_test, A_test, E_test, M_test, N_test, y_test], \
    task_name, transformers = molnet.load_molnet(data=data, split=split, seed=seed)

    if full is False:
        if test is False:
            return ([X_train[::ratio], A_train[::ratio], E_train[::ratio], M_train[::ratio], N_train[::ratio]],
                    y_train[::ratio]), \
                   ([X_valid, A_valid, E_valid, M_valid, N_valid], y_valid)
        else:
            return ([X_train[::ratio], A_train[::ratio], E_train[::ratio], M_train[::ratio], N_train[::ratio]],
                    y_train[::ratio]), \
                   ([X_valid, A_valid, E_valid, M_valid, N_valid], y_valid), \
                   ([X_test, A_test, E_test, M_test, N_test], y_test), \
                   task_name, transformers
    if full is True:
        if test is False:
            return ([X_train, A_train, E_train, M_train, N_train], y_train), \
                   ([X_valid, A_valid, E_valid, M_valid, N_valid], y_valid)
        else:
            return ([X_train, A_train, E_train, M_train, N_train], y_train), \
                   ([X_valid, A_valid, E_valid, M_valid, N_valid], y_valid), \
                   ([X_test, A_test, E_test, M_test, N_test], y_test), \
                   task_name, transformers


def test_load_data():
    load_data(data='qm7', split='stratified', ratio=1)
    load_data(data='qm8', split='random', ratio=2)
    load_data(data='qm9', split='random', ratio=10)
    load_data(data='lipo', split='random', ratio=2)
    load_data(data='esol', split='random', ratio=1)
    load_data(data='freesolv', split='random', ratio=1)


if __name__ == '__main__':
    test_load_data()
