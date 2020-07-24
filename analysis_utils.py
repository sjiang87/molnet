import json
import pickle
import glob
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def to_sec(ts):
    try:
        return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()
    except:
        return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f').timestamp()
    
def three_random_split(DATA_DIR, multi_class=False):
    y_true = []
    y_pred = []
    files = sorted(glob.glob(DATA_DIR + 'best_archs_result_0_*.pickle'))
    for file in files:
        with open(file, 'rb') as f:
            _ = pickle.load(f)
            for _ in range(3):
                if multi_class:
                    y_true.append(pickle.load(f)[np.newaxis, ...])
                    y_pred.append(pickle.load(f).squeeze()[np.newaxis, ...])
                else:
                    y_true.append(pickle.load(f).ravel())
                    y_pred.append(pickle.load(f).ravel().squeeze())
    train_true = np.vstack([y_true[i] for i in [0, 3, 6]])
    train_pred = np.vstack([y_pred[i] for i in [0, 3, 6]])
    valid_true = np.vstack([y_true[i] for i in [1, 4, 7]])
    valid_pred = np.vstack([y_pred[i] for i in [1, 4, 7]])
    test_true = np.vstack([y_true[i] for i in [2, 5, 8]])
    test_pred = np.vstack([y_pred[i] for i in [2, 5, 8]])
    return train_true, train_pred, valid_true, valid_pred, test_true, test_pred

def three_random_mean_std(DATA_DIR, multi_class=False):
    output = three_random_split(DATA_DIR, multi_class=multi_class)
    funcs = [mean_absolute_error, mean_squared_error, r2_score]
    
    if not multi_class:
        result = []
        for func in funcs:
            for i in range(3):
                result.append([func(output[i*2][j], output[i*2+1][j]) for j in range(len(output[0]))])
        result = np.array(result)
        m = result.mean(axis=1)
        s = result.std(axis=1)
        print(tabulate([['Train', f'{m[0]:0.4f}+/-{s[0]:0.4f}', f'{m[3]:0.4f}+/-{s[3]:0.4f}', f'{m[6]:0.4f}+/-{s[6]:0.4f}'], 
                        ['Valid', f'{m[1]:0.4f}+/-{s[1]:0.4f}', f'{m[4]:0.4f}+/-{s[4]:0.4f}', f'{m[7]:0.4f}+/-{s[7]:0.4f}'],
                        ['Test', f'{m[2]:0.4f}+/-{s[2]:0.4f}', f'{m[5]:0.4f}+/-{s[5]:0.4f}', f'{m[8]:0.4f}+/-{s[8]:0.4f}']],
                       headers=['', 'MAE', 'MSE', 'R2']))
    else:
        for c in range(output[0].shape[-1]):
            result = []
            for func in funcs:
                for i in range(3):
                    result.append([func(output[i*2][j, :, c], output[i*2+1][j, :, c]) for j in range(len(train_true))])
            result = np.array(result)
            m = result.mean(axis=1)
            s = result.std(axis=1)
            print(tabulate([['Train', f'{m[0]:0.4f}+/-{s[0]:0.4f}', f'{m[3]:0.4f}+/-{s[3]:0.4f}', f'{m[6]:0.4f}+/-{s[6]:0.4f}'], 
                            ['Valid', f'{m[1]:0.4f}+/-{s[1]:0.4f}', f'{m[4]:0.4f}+/-{s[4]:0.4f}', f'{m[7]:0.4f}+/-{s[7]:0.4f}'],
                            ['Test', f'{m[2]:0.4f}+/-{s[2]:0.4f}', f'{m[5]:0.4f}+/-{s[5]:0.4f}', f'{m[8]:0.4f}+/-{s[8]:0.4f}']],
                           headers=['', 'MAE', 'MSE', 'R2']))
    return m, s

state_dims = ['dim(4)', 'dim(8)', 'dim(16)', 'dim(32)']
Ts = ['repeat(1)', 'repeat(2)', 'repeat(3)', 'repeat(4)']
attn_methods = ['attn(const)', 'attn(gcn)', 'attn(gat)', 'attn(sym-gat)', 'attn(linear)', 'attn(gen-linear)', 'attn(cos)']
attn_heads = ['head(1)', 'head(2)', 'head(4)', 'head(6)']
aggr_methods = ['aggr(max)', 'aggr(mean)', 'aggr(sum)']
update_methods = ['update(gru)', 'update(mlp)']
activations = ['act(sigmoid)', 'act(tanh)', 'act(relu)', 'act(linear)', 'act(elu)', 'act(softplus)', 'act(leaky_relu)', 'act(relu6)']

out = []
for state_dim in state_dims:
    for T in Ts:
        for attn_method in attn_methods:
            for attn_head in attn_heads:
                for aggr_method in aggr_methods:
                    for update_method in update_methods:
                        for activation in activations:
                            out.append([state_dim, T, attn_method, attn_head, aggr_method, update_method, activation])

out_pool = []
for functions in ['GlobalSumPool', 'GlobalMaxPool', 'GlobalAvgPool']:
    for axis in ['(feature)', '(node)']:  # Pool in terms of nodes or features
        out_pool.append(functions+axis)
out_pool.append('flatten')
for state_dim in [16, 32, 64]:
    out_pool.append(f'AttentionPool({state_dim})')
out_pool.append('AttentionSumPool')

out_connect = ['skip', 'connect']

def get_gat(index):
    return out[index]
def get_pool(index):
    return out_pool[index]
def get_connect(index):
    return out_connect[index]

def create_csv(DATA_DIR, data):
    archs = np.array(data['arch_seq'])
    rewards = np.array(data['raw_rewards'])
    a = np.empty((len(archs),0), dtype=np.object)
    a = np.append(a, archs, axis=-1)
    a = np.append(a, rewards[..., np.newaxis], axis=-1)
    b = np.empty((0,29), dtype=np.object)
    for i in range(len(a)):
        temp = a[i, :]
        b0 = [get_gat(temp[0])[i]+'[cell1]' for i in range(len(get_gat(temp[0])))]
        b1 = [get_connect(temp[1])+'[link1]']
        b2 = [get_gat(temp[2])[i]+'[cell2]' for i in range(len(get_gat(temp[2])))]
        b3 = [get_connect(temp[3])+'[link2]']
        b4 = [get_connect(temp[4])+'[link3]']
        b5 = [get_gat(temp[5])[i]+'[cell3]' for i in range(len(get_gat(temp[5])))]
        b6 = [get_connect(temp[6])+'[link4]']
        b7 = [get_connect(temp[7])+'[link5]']
        b8 = [get_connect(temp[8])+'[link6]']
        b9 = [get_pool(temp[9])]
        bout = b0+b1+b2+b3+b4+b5+b6+b7+b8+b9+[temp[10]]
        bout = np.array(bout, dtype=object)
        b = np.append(b, bout[np.newaxis, ...], axis=0)
    table = pd.DataFrame(data=b)
    table.to_csv(DATA_DIR + 'nas_result.csv', encoding='utf-8', index=False, header=False)