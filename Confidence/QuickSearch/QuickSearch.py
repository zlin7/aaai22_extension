try:
    import Confidence.QuickSearch.QuickSearch_cython as cdc
    _CYTHON_ENABLED = True
except:
    try:
        import QuickSearch_cython as cdc
        _CYTHON_ENABLED = True
    except:
        _CYTHON_ENABLED = False
        raise NotImplementedError("This is a warning of potentially slow compute. You could uncomment this line and use the Python implementation instead of Cython.")
import numpy as np
import ipdb
#_CYTHON_ENABLED = False
_MODE_RECALL, _MODE_PRECISION, _MODE_F1, _MODE_NONE = 0, 1, 2, 4
_FLAG_FILLMAX = 2 ** 5
_MODE_MASK = 0xf

_D_LA = 0.
_D_LC = 0.
_D_LAS = 0.
_D_LCS = 0.
_D_PA = 1
_D_PAS = 2
_D_PC = 2
_D_PCS = 2

#_LOSS_PARAMS = (a_star, r_stars, weights, mode, la, lc, lcs, pa, pc, pcs)
def make_loss_params(a_star=0., rks=None, weights=None, mode=_MODE_RECALL,
                     la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS, **kwargs):
    return (a_star, rks, weights, mode, la, las, lc, lcs, pa, pas, pc, pcs)

def one_hot(labels, K=3):
    labels_py = np.zeros((len(labels), K), dtype=np.int)
    for i in range(len(labels)):
        labels_py[i, labels[i]] = 1
    return labels_py


def _parse_mode(mode):
    fill_max = True if (mode & _FLAG_FILLMAX) > 0 else 0
    mode = mode & 0xf
    return fill_max,mode

def _check_flag(mode, flag):
    return (mode & flag) > 0

def thresholding_py(ts, output):
    pred = np.asarray(output > ts, dtype=np.int)
    return pred

def __loss_overall_helper(total_err, total_sure, N, loss_params):
    a_star, r_star, _, _, la, las, lc, lcs, pa, pas, pc, pcs = loss_params
    if r_star is None: r_star = 0.

    a_diff = (1. - total_sure / float(N) - a_star)
    ambiguity_loss = max(a_diff, 0) ** pa
    ambiguity_loss_sim = abs(a_diff) ** pas

    r_diff = total_err / float(max(total_sure, 1)) - r_star
    coverage_loss = np.power(max(r_diff, 0), pc)
    coverage_loss_sim = np.power(abs(r_diff), pcs)
    loss = ambiguity_loss * la + ambiguity_loss_sim * las + coverage_loss * lc + coverage_loss_sim * lcs
    return loss

def __loss_class_specific_complete_helper(TP, FP, FN, total_sure, N, loss_params, to_print=False):
    a_star, r_stars, weights, mode, la, las, lc, lcs, pa, pas, pc, pcs = loss_params
    PredP = TP + FP
    if PredP.min() == 0 or TP.min() == 0: return np.inf

    a_diff = (1 - total_sure / N) - a_star
    a_loss = max(a_diff, 0.) ** pa
    as_loss = abs(a_diff) ** pa

    precision_ = TP / PredP
    recall_ = TP / (TP + FN)
    if mode == _MODE_RECALL:
        r_diff = 1 - recall_
    if mode == _MODE_PRECISION:
        r_diff = 1 - precision_
    if mode == _MODE_F1:
        r_diff = 1 - 2 * (precision_ * recall_) / (precision_ + recall_)
    if r_stars is not None: r_diff -= r_stars
    if weights is None: weights = np.ones(len(TP))
    c_loss = np.dot(np.power(r_diff.clip(0, 1), pc), weights)
    cs_loss = np.dot(np.power(np.abs(r_diff), pcs), weights)

    return a_loss * la + as_loss * las + c_loss * lc + cs_loss * lcs

def loss_overall_py(preds, labels, max_classes=None,
                    fill_max=False,
                    loss_params=None):
    cnt = preds.sum(1)
    cnt1 = np.expand_dims(cnt == 1, 1)
    total_sure = sum(cnt1.squeeze(1))
    risk_indicator = labels * (1 - preds)
    total_err = (risk_indicator * cnt1).sum()
    if fill_max:
        cnt0 = cnt == 0
        total_sure += sum(cnt0)
        new_risk = max_classes != np.argmax(labels, 1)
        total_err += sum(cnt0 & new_risk)
    return __loss_overall_helper(total_err, total_sure, len(labels), loss_params)

def loss_class_specific_py(preds, labels, max_classes,
                           fill_max=False,
                           loss_params=None):
    N,K = preds.shape
    cnt = preds.sum(1)
    sure_msk = cnt == 1
    total_sure = np.sum(sure_msk)
    correct_msk = np.sum(preds * labels, 1) == 1
    TP = np.sum(labels[sure_msk & correct_msk], 0)
    FN = np.sum(labels[sure_msk & ~correct_msk], 0)
    FP = np.sum(preds[sure_msk & ~correct_msk], 0)
    if fill_max:
        sure_msk = cnt == 0
        total_sure += np.sum(sure_msk)
        correct_msk = max_classes == np.argmax(labels, 1)
        TP += np.sum(labels[sure_msk & correct_msk], 0)
        FN += np.sum(labels[sure_msk & ~correct_msk], 0)
        FP += np.sum(preds[sure_msk & ~correct_msk], 0)

    return __loss_class_specific_complete_helper(TP, FP, FN, total_sure, N, loss_params)


def __update_cnts(pred, y, TP, FP, FN, inc=1):
    if pred == y:
        TP[y] += inc
    else:
        FP[pred] += inc
        FN[y] += inc
    return inc

def search_full_class_specific_py(mo, rnkscores_or_maxclasses, scores_idx, labels, ps, d,
                                   fill_max=False,
                                  loss_params=None,
                                  ):
    """

    :param mo: When rnkscores_or_maxclasses is max_classes, mo should be idx2rnk (namely, the score IS the rank)
    :param rnkscores_or_maxclasses:
    :param scores_idx:
    :param labels:
    :param rks:
    :param class_weights:
    :param ps:
    :param d:
    :param la:
    :param lc:
    :param lcs:
    :param mode:
    :param fill_max:
    :return:
    """
    N,K = mo.shape
    ps = ps.copy()
    if len(rnkscores_or_maxclasses.shape) == 2:
        ts = [(rnkscores_or_maxclasses[ps[ki], ki]) for ki in range(K)]
        max_classes = np.argmax(mo, 1)
    else:
        ts = [ps[ki] for ki in range(K)]
        max_classes = rnkscores_or_maxclasses

    preds = thresholding_py(ts, mo)
    preds[:, d] = 1
    cnt = preds.sum(1)
    labels_onehot = one_hot(labels, K)
    cnt1_msk = cnt == 1
    correct_msk = cnt1_msk & (np.sum(preds * labels_onehot, 1) == 1)
    total_sure = np.sum(cnt1_msk)
    TP = np.sum(labels_onehot[cnt1_msk & correct_msk], 0)
    FN = np.sum(labels_onehot[cnt1_msk & ~correct_msk], 0)
    FP = np.sum(preds[cnt1_msk & ~correct_msk], 0)
    if fill_max:
        cnt0_msk = cnt == 0
        cnt0_correct_msk = max_classes == labels
        total_sure += np.sum(cnt0_msk)
        TP += np.sum(labels_onehot[cnt0_msk & cnt0_correct_msk], 0)
        FN += np.sum(labels_onehot[cnt0_msk & ~cnt0_correct_msk], 0)
        FP += np.sum(preds[cnt0_msk & ~cnt0_correct_msk], 0)
    #print(TP, FP, FN)
    _preds = np.ones(N, dtype=int) * K
    _preds[cnt1_msk] = np.argmax(preds[cnt1_msk], 1)
    cnt2_msk = np.asarray((cnt == 2) & (preds[:, d]), dtype=bool) #second mask is trivial
    preds[:, d] = 0 #this order cannot chance
    _preds[cnt2_msk] = np.argmax(preds[cnt2_msk], 1)

    best_i, best_loss = -1, np.inf
    for ijjj in range(N-1):
        i = scores_idx[ijjj, d]
        yi = labels[i]
        tint = _preds[i]
        if cnt[i] == 1 or cnt[i] == 2:
            total_sure += __update_cnts(tint, yi, TP, FP, FN, 1 if cnt[i] == 2 else -1)
            if fill_max and cnt[i] == 1:
                total_sure += __update_cnts(max_classes[i], yi, TP, FP, FN, 1)
        cnt[i] -= 1
        curr_loss = __loss_class_specific_complete_helper(TP, FP, FN, total_sure, N, loss_params)
        if curr_loss < best_loss:
            best_i, best_loss = ijjj, curr_loss
            #__loss_class_specific_complete_helper(TP, FP, FN, total_sure, N, loss_params, to_print=False)
    return best_i, best_loss ##, l1, l2


def search_full_overall_py(mo, rnkscores_or_maxclasses, scores_idx, labels, ps, d,
                           fill_max=False, loss_params=None):
    N,K = mo.shape
    ps = ps.copy()
    if len(rnkscores_or_maxclasses.shape) == 2:
        ts = [(rnkscores_or_maxclasses[ps[ki], ki]) for ki in range(K)]
        max_classes = np.argmax(mo, 1)
    else:
        ts = [ps[ki] for ki in range(K)]
        max_classes = rnkscores_or_maxclasses
    preds = thresholding_py(ts, mo)
    preds[:, d] = 1
    cnt = preds.sum(1)

    #initialize mems for ijjj=0
    total_err, total_sure = 0, 0
    for ii in range(N):
        yii = labels[ii]
        if cnt[ii] == 1:
            total_sure += 1
            total_err += 1 - preds[ii, yii]
        if cnt[ii] == 0 and fill_max:
            total_sure += 1
            if max_classes[ii] != yii: total_err += 1
    best_i, best_loss = -1, np.inf

    for ijjj in range(N-1):
        i = scores_idx[ijjj, d]
        yi = labels[i]
        #preds[i, d] will change from 1 to 0
        if cnt[i] == 2:#unsure -> sure
            if d == yi:#we miss this item in this case, so error increases by 1
                total_err += 1
            elif mo[i, yi] <= ts[yi]: #we miss the item
                total_err += 1
            total_sure += 1
        elif cnt[i] == 1: #Sure -> unsure. Also, this cnt has to be class d
            if d != yi: total_err -= 1
            total_sure -= 1
            if fill_max:
                total_sure += 1
                if max_classes[i] != yi: total_err += 1
        else: #Do nothing as we have an unsure -> unsure case
            pass
        cnt[i] -= 1

        curr_loss = __loss_overall_helper(total_err, total_sure, N, loss_params)

        if curr_loss < best_loss:
            best_i, best_loss = ijjj, curr_loss
    return best_i, best_loss

def coord_descnet_class_specific_py(mo, rnkscores_or_maxclasses, scores_idx, labels, ps,
                                    fill_max=False, loss_params=None):
    N, K = mo.shape
    #if fill_max: mode += _FLAG_FILLMAX
    best_loss, ps = np.inf, ps.copy()

    keep_going = True

    while keep_going:
        curr_ps = np.zeros(K)
        best_ki, curr_loss = None, best_loss
        for ki in range(K):
            curr_ps[ki], temp_loss = search_full_class_specific_py(mo, rnkscores_or_maxclasses, scores_idx, labels, ps, ki,
                                                                   fill_max, loss_params)
            if temp_loss < curr_loss:
                best_ki, curr_loss = ki, temp_loss
        if curr_loss < best_loss:
            ps[best_ki] = curr_ps[best_ki]
            best_loss = curr_loss
        else:
            keep_going = False
    return best_loss, ps


def coord_descnet_overall_py(mo, rnkscores_or_maxclasses, scores_idx, labels, ps,
                             fill_max=False, loss_params=None):
    N, K = mo.shape
    best_loss, ps = np.inf, ps.copy()

    keep_going = True

    while keep_going:
        curr_ps = np.zeros(K)
        best_ki, curr_loss = None, best_loss
        for ki in range(K):
            curr_ps[ki], temp_loss = search_full_overall_py(mo, rnkscores_or_maxclasses, scores_idx, labels, ps, ki,
                                                            fill_max=fill_max, loss_params=loss_params)
            if temp_loss < curr_loss:
                best_ki, curr_loss = ki, temp_loss
        if curr_loss < best_loss:
            ps[best_ki] = curr_ps[best_ki]
            best_loss = curr_loss
        else:
            keep_going = False
    return best_loss, ps

#================================
def _global_rnk_search():
    pass

#=================================Interface
#
def loss_overall(idx2rnk, rnk2idx, labels, max_classes, ps,
                 fill_max=False,
                 a_star=0., r=0.3,
                 la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    if not _CYTHON_ENABLED:
        preds = np.asarray(idx2rnk > ps, np.int)
        loss_params = make_loss_params(a_star, r, None, _MODE_NONE, la, las, lc, lcs, pa, pas, pc, pcs)
        return loss_overall_py(preds, one_hot(labels, idx2rnk.shape[1]), max_classes, fill_max, loss_params)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    ps = np.asarray(ps, np.int)
    return cdc.loss_overall_q_(idx2rnk, rnk2idx, labels, max_classes, ps, fill_max,
                               a_star, r, la, las, lc, lcs, pa=pa, pas=pas, pc=pc, pcs=pcs)


def loss_class_specific(idx2rnk, rnk2idx, labels, max_classes, ps,
                        fill_max=False,
                        a_star=0., rks=None, class_weights=None,
                        mode=_MODE_RECALL,
                        la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    if not _CYTHON_ENABLED:
        preds = np.asarray(idx2rnk > ps, np.int)
        loss_params = make_loss_params(a_star, rks, class_weights, mode, la, las, lc, lcs, pa, pas, pc, pcs)
        return loss_class_specific_py(preds, one_hot(labels, idx2rnk.shape[1]), max_classes, fill_max, loss_params)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    ps = np.asarray(ps, np.int)
    if class_weights is not None: class_weights = np.asarray(class_weights, np.float)
    if rks is not None: rks = np.asarray(rks, np.float)
    return cdc.loss_class_specific_q_(idx2rnk, rnk2idx, labels, max_classes, ps,
                                      fill_max,
                                      a_star, rks, class_weights,
                                      mode, la, las, lc, lcs, pa, pas, pc, pcs)

def search_overall(idx2rnk, rnk2idx, labels, max_classes, ps, k,
                   fill_max=False,
                   a_star=0., r=0.3,
                   la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    if not _CYTHON_ENABLED:
        loss_params = make_loss_params(a_star, r, None, _MODE_NONE, la, las, lc, lcs, pa, pas, pc, pcs)
        return search_full_overall_py(idx2rnk, max_classes, rnk2idx, labels, ps, k, fill_max, loss_params)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    ps = np.asarray(ps, np.int)
    return cdc.search_full_overall(idx2rnk, rnk2idx, labels, max_classes, ps, k, fill_max,
                                   a_star, r, la, las, lc, lcs, pa=pa, pas=pas, pc=pc, pcs=pcs)

def search_classSpec(idx2rnk, rnk2idx, labels, max_classes, ps, k,
                     fill_max=False,
                     a_star=0., rks=None, class_weights=None,
                     mode=_MODE_RECALL,
                     la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    if not _CYTHON_ENABLED:
        loss_params = make_loss_params(a_star, rks, class_weights, mode, la, las, lc, lcs, pa, pas, pc, pcs)
        return search_full_class_specific_py(idx2rnk, max_classes, rnk2idx, labels, ps, k,
                                             fill_max, loss_params)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    ps = np.asarray(ps, np.int)
    if rks is not None: rks = np.asarray(rks, np.float)
    return cdc.search_full_class_specific(idx2rnk, rnk2idx, labels, max_classes, ps, k,
                                          fill_max,
                                          a_star, rks, class_weights,
                                          mode, la, las, lc, lcs, pa, pas, pc, pcs)

def coordDescent_overall(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                         fill_max=False, max_step=None,
                         a_star=0., r=0.3,
                         la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    if not _CYTHON_ENABLED:
        assert max_step is None
        loss_params = make_loss_params(a_star, r, None, _MODE_NONE, la, las, lc, lcs, pa, pas, pc, pcs)
        return coord_descnet_overall_py(idx2rnk, max_classes, rnk2idx, labels, init_ps, fill_max, loss_params)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    init_ps = np.asarray(init_ps, np.int)
    return cdc.main_coord_descent_overall_(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                                           max_step, fill_max,
                                           a_star, r, la, las, lc, lcs, pa=pa, pas=pas, pc=pc, pcs=pcs)


def coordDescent_classSpec(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                           fill_max=False, max_step=None,
                           a_star=0., rks=None, class_weights=None,
                           mode=_MODE_RECALL,
                           la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    if not _CYTHON_ENABLED:
        assert max_step is None
        loss_params = make_loss_params(a_star, rks, class_weights, mode, la, las, lc, lcs, pa, pas, pc, pcs)
        return coord_descnet_class_specific_py(idx2rnk, max_classes, rnk2idx, labels, init_ps, fill_max, loss_params)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    init_ps = np.asarray(init_ps, np.int)
    if class_weights is not None: class_weights = np.asarray(class_weights, np.float)
    if rks is not None: rks = np.asarray(rks, np.float)
    return cdc.main_coord_descent_class_specific_(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                                                  max_step, fill_max,
                                                  a_star, rks, class_weights,
                                                  mode, la, las, lc, lcs, pa, pas, pc, pcs)


def coordDescent_classSpec_global(rnk2ik, labels, max_classes, ascending=False,  #input
                                  fill_max=False,  #search
                                  a_star = 0., rks = None, class_weights=None,  #targets and weights
                                  mode=_MODE_RECALL,  #mode
                                  la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS): #details
    assert _CYTHON_ENABLED, "This function currently only has cython version"
    rnk2ik = np.asarray(rnk2ik, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    if isinstance(rks, float): rks = np.asarray([rks] * (len(rnk2ik) // len(labels)))
    assert len(rnk2ik) % len(labels) == 0
    if rks is not None: rks = np.asarray(rks, np.float)
    return cdc.main_coord_descent_class_specific_globalt_(rnk2ik, labels, max_classes, ascending, fill_max,
                                                          a_star, rks, class_weights,
                                                          mode, la, las, lc, lcs, pa, pas, pc, pcs)