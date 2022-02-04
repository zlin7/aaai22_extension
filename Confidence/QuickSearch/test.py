try:
    import QuickSearch as quicksearch
except Exception as err:
    print(err)
    import QuickSearch.QuickSearch as quicksearch
from importlib import reload
reload(quicksearch)

import pandas as pd
import numpy as np
import timeit
import ipdb

def sort_outputs(out):
    scores = np.sort(out, axis=0)
    scores_idx = np.argsort(out, axis=0)
    idx2rnk = np.asarray(pd.DataFrame(out).rank(ascending=True), np.int) - 1
    return scores, scores_idx, idx2rnk

one_hot = quicksearch.one_hot

def get_data():
    np.random.seed(7)
    out = np.random.uniform(0, 1, (1000, 3))
    labels = np.random.randint(0, 3, 1000)
    #ps = np.random.randint(0, 1000, 3)
    ps = np.asarray([556,536, 529])
    scores, scores_idx, idx2rnk = sort_outputs(out)
    ts = np.asarray([scores[ps[i], i] for i in range(3)])
    return ts, out, labels, ps, scores, scores_idx, idx2rnk


def test(f_py, f_c, repeat, desc):
    t_py, t_c = timeit.timeit(f_py, number=repeat), timeit.timeit(f_c, number=repeat)
    print(f"{desc} Time: {t_py:.5f} vs {t_c:.5f}, mult={t_py/t_c:.2f}")
    print(f_py(), f_c())
    print()


if __name__ == '__main__':
    ts, out, labels, ps, scores, rnk2idx, idx2rnk = get_data()
    max_classes = np.asarray(np.argmax(out, axis=1), np.int)
    rnk2idx = np.asarray(rnk2idx, dtype=np.int)
    labels_py = one_hot(labels, 3)
    rks = np.asarray([0.3, 0.3, 0.3], dtype=np.float)

    fill_max = False

    pred = np.asarray(out > ts, dtype=np.int)
    loss_kwargs = {'fill_max': fill_max,
                   'a_star': 1.,
                   'rks': 0.3,
                   'la': 5.03, 'lc': 10, 'lcs': 1.01,
                   'pa': 4, 'pc': 3, 'pcs': 2}

    new_kwargs = {'fill_max': loss_kwargs.get('fill_max'),
                  'loss_params': quicksearch.make_loss_params(**loss_kwargs)}
    loss_kwargs['r'] = loss_kwargs.pop('rks')
    test(lambda: quicksearch.loss_overall_py(pred, labels_py, max_classes, **new_kwargs),
         lambda: quicksearch.loss_overall(idx2rnk, rnk2idx, labels, max_classes, ps, **loss_kwargs),
         repeat=0, desc='Loss Overall')

    test(lambda: quicksearch.search_full_overall_py(out, scores, rnk2idx, labels, ps=ps, d=0, **new_kwargs),
         lambda: quicksearch.search_overall(idx2rnk, rnk2idx, labels, max_classes, ps, 0, **loss_kwargs),
         repeat=0, desc='Search Overall')

    test(lambda: quicksearch.coord_descnet_overall_py(out, scores, rnk2idx, labels, ps, **new_kwargs),
         lambda: quicksearch.coordDescent_overall(idx2rnk, rnk2idx, labels, max_classes, ps, **loss_kwargs),
         repeat=0, desc='Full Run Overall')
    loss_kwargs.pop('r', None)

    loss_kwargs.update({"mode": 1, 'class_weights': None, 'rks': rks, 'a_star': 0.2})
    #ps = np.asarray([int(701), 8, 424], np.int)
    new_kwargs = {'fill_max': loss_kwargs.get('fill_max'), 'loss_params': quicksearch.make_loss_params(**loss_kwargs)}
    test(lambda: quicksearch.loss_class_specific_py(pred, labels_py, max_classes, **new_kwargs),
         lambda: quicksearch.loss_class_specific(idx2rnk, rnk2idx, labels, max_classes, ps, **loss_kwargs),
         repeat=0, desc='Loss ClassSpec')

    test(lambda: quicksearch.search_full_class_specific_py(out, scores, rnk2idx, labels, ps=ps, d=0, **new_kwargs),
         lambda: quicksearch.search_classSpec(idx2rnk, rnk2idx, labels, max_classes, ps=ps, k=0, **loss_kwargs),
         repeat=0, desc='Search')

    test(lambda: quicksearch.coord_descnet_class_specific_py(out, scores, rnk2idx, labels, ps, **new_kwargs),
         lambda: quicksearch.coordDescent_classSpec(idx2rnk, rnk2idx, labels, max_classes, ps, **loss_kwargs),
         repeat=0, desc='Full Run')
