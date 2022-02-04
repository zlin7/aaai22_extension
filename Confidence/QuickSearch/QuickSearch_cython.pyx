#https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
#https://cython-docs2.readthedocs.io/en/latest/src/tutorial/numpy.html
from libc.stdlib cimport calloc, free
#from libcpp cimport bool

import numpy as np
cimport numpy as np
DTYPE = np.float
DTYPE_INT = np.int
_MODE_RECALL, _MODE_PRECISION, _MODE_F1, _MODE_NONE = 0, 1, 2, 3
_FLAG_FILLMAX = 2 ** 5
_MODE_MASK = 0xf

_D_LA = 0.03
_D_LAS = 0.
_D_LC = 10.
_D_LCS = 0.01
_D_PA = 1
_D_PAS = 2
_D_PC = 2
_D_PCS = 2

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t DTYPE_int_t
cdef struct loss_param_struct:
    double a_star #targets
    double r_star
    double* r_stars #targets
    double* weights
    int    mode
    double la
    double las
    double lc
    double lcs
    int    pa
    int    pas
    int    pc
    int    pcs

cdef _parse_mode(int mode):
    cdef int fill_max = <int> ((mode & _FLAG_FILLMAX) > 0)
    mode = mode & _MODE_MASK
    return fill_max,mode

cdef int _check_flag(int mode, int flag):
    return <int> ((mode & flag) == flag)

cdef int _update_counts(int pred, int truth, int* TP, int* FP, int* FN, int inc):
    if pred == truth:
        TP[truth] += inc
    else:
        FP[pred] += inc
        FN[truth] += inc
    return inc
#============================Compute

cdef double loss_overall_helper__(int total_err, int total_sure, int N, loss_param_struct params):
    #if total_sure == 0: return np.inf
    cdef tempf = 1. - total_sure / <double> N - params.a_star
    cdef double a_loss = max(tempf, 0.) ** params.pa
    cdef double as_loss = abs(tempf) ** params.pas

    tempf = total_err / <double> max(total_sure,1) - params.r_star
    cdef double c_loss = max(tempf, 0.) ** params.pc
    cdef double cs_loss = abs(tempf) ** params.pcs
    return a_loss * params.la + as_loss * params.las + c_loss * params.lc + cs_loss * params.lcs

cdef double loss_class_specific_complete_helper__(int K, int* TP, int* FP, int* FN,
                                                  int total_sure, int N,
                                                  loss_param_struct params):
    #If rs is NULL, simply penalize (it's a weighted risk loss)
    #If rs is not NULL, they are the targets..
    cdef double precision_, recall_, f1_, a_diff, tf, weight_k=1., c_loss=0., cs_loss = 0., loss_

    a_diff = 1. - total_sure / <double>N - params.a_star
    cdef double a_loss = max(a_diff, 0.) ** params.pa
    cdef double as_loss = abs(a_diff) ** params.pas

    for ki in range(K):
        if TP[ki] + FP[ki] == 0 or TP[ki] == 0: return np.inf
        precision_ = TP[ki] / <double> (TP[ki] + FP[ki])
        recall_ = TP[ki] / <double> (TP[ki] + FN[ki])
        if (params.mode & _MODE_MASK) == _MODE_PRECISION: tf = 1 - precision_
        if (params.mode & _MODE_MASK) == _MODE_RECALL: tf = 1 - recall_
        if (params.mode & _MODE_MASK) == _MODE_F1: tf = 1 - 2 * (precision_ * recall_) / (precision_ + recall_)
        if params.r_stars != NULL: tf -= params.r_stars[ki]
        if params.weights != NULL: weight_k = params.weights[ki]
        c_loss += weight_k * (max(tf, 0.) ** params.pc)
        cs_loss += weight_k * (abs(tf) ** params.pcs)

    loss_ = a_loss * params.la + as_loss * params.las + c_loss * params.lc + cs_loss * params.lcs
    return loss_

#=============================Searches
cdef int __full_eval(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk, np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,
                     np.ndarray[DTYPE_int_t, ndim=1] labels, np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                     np.ndarray[DTYPE_int_t, ndim=1] ps,
                     int* TP, int* FP, int* FN,
                     int fill_max):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, cnt, total_sure = 0, temp_last_pred
    for ni in range(N):
        yi = labels[ni]
        cnt = temp_last_pred = 0
        for ki in range(K):
            if idx2rnk[ni, ki] > ps[ki]:
                cnt += 1
                pred_i = ki
        if cnt == 1:
            total_sure += _update_counts(pred_i, yi, TP, FP, FN, 1)
        if cnt == 0 and fill_max:
            total_sure += _update_counts(max_classes[ni], yi, TP, FP, FN, 1)
    return total_sure

cdef (int, int, int*) __full_eval_overall(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk, np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,
                                    np.ndarray[DTYPE_int_t, ndim=1] labels, np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                    np.ndarray[DTYPE_int_t, ndim=1] ps,
                                    int fill_max):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, cnt_i
    cdef int total_err = 0, total_sure = 0
    cdef int* cnt = <int*> calloc(N, sizeof(int))
    for ni in range(N):
        yi = labels[ni]
        cnt_i = 0
        for ki in range(K):
            if idx2rnk[ni, ki] > ps[ki]:
                cnt_i += 1
        if cnt_i == 1:
            total_sure += 1
            if idx2rnk[ni, yi] <= ps[yi]:
                total_err += 1
        elif cnt_i == 0 and fill_max:
            total_sure += 1
            if max_classes[ni] != yi:
                total_err += 1
        cnt[ni] = cnt_i
    return total_err, total_sure, cnt

cdef double loss_class_specific_q__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                    np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                    np.ndarray[DTYPE_int_t, ndim=1] labels,
                                    np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                   #Start of memory stuff
                                   np.ndarray[DTYPE_int_t, ndim=1] ps,  #positions
                                   #Start of loss function args
                                    loss_param_struct lp):

    cdef int fill_max = _check_flag(lp.mode, _FLAG_FILLMAX)

    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, cnt, temp_last_pred
    cdef int* TP = <int*> calloc(K, sizeof(int))
    cdef int* FP = <int*> calloc(K, sizeof(int))
    cdef int* FN = <int*> calloc(K, sizeof(int)) #sure_cnt = TP+FP
    cdef int total_sure = __full_eval(idx2rnk, rnk2idx, labels, max_classes, ps, TP, FP, FN, fill_max)
    cdef double loss = loss_class_specific_complete_helper__(K, TP, FP, FN, total_sure, N, lp)
    free(TP); free(FP); free(FN)
    return loss


cdef double loss_overall_q__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #input data args
                             np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,
                             np.ndarray[DTYPE_int_t, ndim=1] labels,
                             np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                             #Start of memory stuff
                             np.ndarray[DTYPE_int_t, ndim=1] ps,  #positions
                             #Start of loss function args
                             loss_param_struct lp):
    cdef int fill_max = _check_flag(lp.mode, _FLAG_FILLMAX)
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int total_err = 0, total_sure = 0
    cdef int* cnt = NULL
    total_err, total_sure, cnt = __full_eval_overall(idx2rnk, rnk2idx, labels, max_classes, ps, fill_max)
    cdef double loss_ = loss_overall_helper__(total_err, total_sure, N, lp)
    free(cnt)
    return loss_

cdef (int, double) search_full_class_specific_complete_globalt__(np.ndarray[DTYPE_int_t, ndim=2] rnk2ik, #rnk to i and k. Rank is descending
                                                                 np.ndarray[DTYPE_int_t, ndim=1] labels,
                                                                 np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                                                 int ascending,
                                                                 #Start of loss function args
                                                                 loss_param_struct lp
                                                                 ):
    cdef int fill_max = _check_flag(lp.mode, _FLAG_FILLMAX)

    cdef int best_rnk, rnk, total = rnk2ik.shape[0], ni, ki, total_sure = 0, tempint, N = labels.shape[0], K, yi
    K = total / N
    cdef double best_loss = np.inf, tempf
    cdef char** preds = <char**> calloc(K, sizeof(char*)) #preds[k, i]
    for ki in range(K): preds[ki] = <char*> calloc(N, sizeof(char))
    cdef int* TP = <int*> calloc(K, sizeof(int))
    cdef int* FP = <int*> calloc(K, sizeof(int))
    cdef int* FN = <int*> calloc(K, sizeof(int)) #sure_cnt = TP+FP
    cdef int* cnt = <int*> calloc(N, sizeof(int)) #This does not count fill_max

    if fill_max:
        for ni in range(N):
            tempint = max_classes[ni]
            #preds[tempint][ni] = 1
            total_sure += _update_counts(tempint, labels[ni], TP, FP, FN, 1)

    for rnk in range(total):
        if ascending: rnk = total - rnk - 1 #rnk better be increasing
        ni = rnk2ik[rnk, 0]
        ki = rnk2ik[rnk, 1]
        yi = labels[ni]
        #eval...
        if cnt[ni] == 0: #0->1
            if fill_max:
                total_sure += _update_counts(max_classes[ni], yi, TP, FP, FN, -1)
            total_sure += _update_counts(ki, yi, TP, FP, FN, 1)
        elif cnt[ni] == 1: #1->2
            for tempint in range(K):
                if preds[tempint][ni] == 1:
                    total_sure += _update_counts(tempint, yi, TP, FP, FN, -1)
                    break
        preds[ki][ni] = 1
        cnt[ni] += 1

        curr_loss = loss_class_specific_complete_helper__(K, TP, FP, FN, total_sure, N, lp)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_rnk = rnk + 1
    free(TP); free(FP); free(FN);
    for ki in range(K): free(preds[ki])
    free(preds); free(cnt)
    return best_rnk, best_loss

cdef (int, double) search_full_class_specific_complete__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                                         np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                                         np.ndarray[DTYPE_int_t, ndim=1] labels,
                                                         np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                                         #Start of memory stuff
                                                         np.ndarray[DTYPE_int_t, ndim=1] ps,  #positions
                                                         int dim, int start_pos, int end_pos,
                                                         #Start of loss function args
                                                         loss_param_struct lp):
    cdef int fill_max = _check_flag(lp.mode, _FLAG_FILLMAX)

    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, tempint, tempint2, total_sure = 0, orig_pd = ps[dim]
    if end_pos == -1: end_pos = N - 1 #Update  the actual end_pos. [start_pos, end_pos)
    ps[dim] = start_pos - 1 #inclusive of start_pos, so we first eval the loss for start_pos-1
    cdef int* TP = <int*> calloc(K, sizeof(int))
    cdef int* FP = <int*> calloc(K, sizeof(int))
    cdef int* FN = <int*> calloc(K, sizeof(int)) #sure_cnt = TP+FP
    cdef int* cnt = <int*> calloc(N, sizeof(int))
    cdef int* alt = <int*> calloc(N, sizeof(int))
    #cnt[i] == 1: alt[i] = k means the sure prediction is k
    #cnt[i] == 2: alt[i] = k means this is the prediction other than k=dim, >=K means both are not..
    #otherwise, alt[i] does not mean anything

    for ni in range(N):
        tempint = 0 #tempint counts how many k!=dim in prediction. alt[ni] is the sum of all of them
        yi = labels[ni]
        for ki in range(K):
            if idx2rnk[ni, ki] > ps[ki]:
                cnt[ni] += 1
                if ki != dim:
                    tempint += 1
                    alt[ni] += ki
        if cnt[ni] == 1:
            if tempint == 0: alt[ni] = dim #alt[ni] is now the prediction, yi is truth
            total_sure += _update_counts(alt[ni], yi, TP, FP, FN, 1)
        elif cnt[ni] == 2:
            if tempint == 2: alt[ni] = K #Both are not dim, so it stays uncertain regardless of the threshold of dim
        elif cnt[ni] == 0 and fill_max: #Don't worry about this case later, as cnt will only decrease as we search
            total_sure += _update_counts(max_classes[ni], yi, TP, FP, FN, 1)
    #print(TP[0], TP[1], TP[2], "|", FP[0], FP[1], FP[2], "|", FN[0], FN[1], FN[2])

    #iterate through all quantiles
    cdef double curr_loss, best_loss = np.inf
    cdef int best_nj = -1, nj
    for nj in range(start_pos, end_pos):#At nj, we are computing the loss for the case >nj
        ni = rnk2idx[nj, dim]
        yi = labels[ni]
        tempint = alt[ni] #prediction (when we do use this)
        if cnt[ni] == 2: #unsure->sure
            #assert tempint != K and tempint != dim
            total_sure += _update_counts(tempint, yi, TP, FP, FN, 1)
        elif cnt[ni] == 1: #sure -> unsure
            total_sure += _update_counts(tempint, yi, TP, FP, FN, -1)
            if fill_max:#unsure -> sure, but uses the alternative prediction in max_classes..
                total_sure += _update_counts(max_classes[ni], yi, TP, FP, FN, 1)
                #cnt[ni] += 1 #We will use cnt only to count the number of triggered thresholds for clarity
        cnt[ni] -= 1
        curr_loss = loss_class_specific_complete_helper__(K, TP, FP, FN, total_sure, N, lp)
        if curr_loss < best_loss:
            best_nj, best_loss = nj, curr_loss
    free(TP); free(FP); free(FN);
    free(cnt); free(alt);
    ps[dim] = orig_pd
    return best_nj, best_loss

cdef (int, double) search_full_overall__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                         np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                         np.ndarray[DTYPE_int_t, ndim=1] labels,
                                         np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                         #Start of memory stuff
                                         np.ndarray[DTYPE_int_t, ndim=1] ps,  #positions
                                         int dim, int start_pos, int end_pos,
                                         #Start of loss function args
                                         loss_param_struct lp):
    cdef int fill_max = _check_flag(lp.mode, _FLAG_FILLMAX)

    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, tempint, orig_pd = ps[dim]
    if end_pos == -1: end_pos = N - 1 #Update  the actual end_post. [start_pos, end_pos)
    ps[dim] = start_pos - 1 #inclusive of start_pos, so we first eval the loss for start_pos-1
    cdef int* cnt = NULL
    cdef int total_err = 0, total_sure = 0
    total_err, total_sure, cnt = __full_eval_overall(idx2rnk, rnk2idx, labels, max_classes, ps, fill_max)

    cdef double curr_loss, best_loss = np.inf
    cdef int best_nj = -1, nj
    for nj in range(start_pos, end_pos): #At nj, we are computing the loss for the case >nj
        ni = rnk2idx[nj, dim]
        yi = labels[ni]
        if cnt[ni] == 2: #unsure->sure
            if dim == yi: #We miss this class dim item
                total_err += 1
            elif idx2rnk[ni, yi] <= ps[yi]: #We originally missed this but now this becomes sure
                total_err += 1
            total_sure += 1
        elif cnt[ni] == 1: #sure -> unsure
            if dim != yi: total_err -= 1 #only in this case was there an error before this change
            total_sure -= 1
            if fill_max:
                total_sure += 1
                if max_classes[ni] != yi: total_err += 1
        cnt[ni] -= 1
        curr_loss = loss_overall_helper__(total_err, total_sure, N, lp)
        if curr_loss < best_loss:
            best_nj, best_loss = nj, curr_loss

    free(cnt)
    ps[dim] = orig_pd
    return best_nj, best_loss

cdef (int, double) main_coord_descent_class_specific__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                                np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                                np.ndarray[DTYPE_int_t, ndim=1] labels,
                                                np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                                np.ndarray[DTYPE_int_t, ndim=1] init_ps, #initial positions
                                                int* mod_ps, #Store ps here
                                                double* best_loss_ptr,int max_step,
                                                       loss_param_struct lp):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef double best_loss = np.inf, curr_loss, tempf
    cdef int keep_going = 1, ki, pi, curr_best_ki, start_pos, end_pos, max_step_temp=-1, n_searches = 0
    cdef np.ndarray[DTYPE_int_t, ndim=1] ps = init_ps.copy()
    cdef np.ndarray[DTYPE_int_t, ndim=1] curr_p = np.empty([K], dtype=DTYPE_INT)
    while keep_going == 1:
        n_searches += 1
        curr_loss = best_loss
        for ki in range(K):
            if max_step > 0:
                start_pos, end_pos = max(ps[ki]-max_step, 0), min(ps[ki]+max_step, N-1)
            else:
                start_pos, end_pos = 0, -1
            curr_p[ki], tempf = search_full_class_specific_complete__(idx2rnk, rnk2idx, labels, max_classes, ps, ki, start_pos, end_pos,
                                                                      lp)
            if tempf < curr_loss:
                curr_loss, curr_best_ki = tempf, ki
        if curr_loss < best_loss:
            ps[curr_best_ki] = curr_p[curr_best_ki]
            best_loss = curr_loss
            if max_step_temp > 0: #Switch back as we used this to move further in this round
                max_step, max_step_temp = max_step_temp, -1
        else:
            if max_step_temp == -1 and max_step > 0: #We could try moving further
                max_step_temp, max_step = max_step, -1
            else:
                keep_going = 0

    for ki in range(K):
        mod_ps[ki] = ps[ki]
    best_loss_ptr[0] = best_loss
    return (n_searches, best_loss)

cdef (int, double) main_coord_descent_overall__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                         np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                         np.ndarray[DTYPE_int_t, ndim=1] labels,
                                         np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                         np.ndarray[DTYPE_int_t, ndim=1] init_ps,  #initial positions
                                         int* mod_ps,  #Store ps here
                                         double* best_loss_ptr,
                                                int max_step,
                                                loss_param_struct lp):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef double best_loss = np.inf, curr_loss, tempf
    cdef int keep_going = 1, ki, pi, curr_best_ki, start_pos, end_pos, max_step_temp=-1, n_searches = 0
    cdef np.ndarray[DTYPE_int_t, ndim=1] ps = init_ps.copy()
    cdef np.ndarray[DTYPE_int_t, ndim=1] curr_p = np.empty([K], dtype=DTYPE_INT)
    while keep_going == 1:
        n_searches += 1
        curr_loss = best_loss
        for ki in range(K):
            if max_step > 0:
                start_pos, end_pos = max(ps[ki]-max_step, 0), min(ps[ki]+max_step, N-1)
            else:
                start_pos, end_pos = 0, -1
            curr_p[ki], tempf = search_full_overall__(idx2rnk, rnk2idx, labels, max_classes, ps, ki, start_pos, end_pos, lp)
            if tempf < curr_loss:
                curr_loss, curr_best_ki = tempf, ki
        if curr_loss < best_loss:
            ps[curr_best_ki] = curr_p[curr_best_ki]
            best_loss = curr_loss
            if max_step_temp > 0: #Switch back as we used this to move further in this round
                max_step, max_step_temp = max_step_temp, -1
        else:
            if max_step_temp == -1 and max_step > 0: #We could try moving further
                max_step_temp, max_step = max_step, -1
            else:
                keep_going = 0
    for ki in range(K):
        mod_ps[ki] = ps[ki]
    best_loss_ptr[0] = best_loss
    return (n_searches, best_loss)

#=============================Python Interfaces
#https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python
cpdef int _encode_mode(target='recall', fill_max=False):
    try:
        base_mode = {'recall': _MODE_RECALL, 'f1': _MODE_F1, 'precision': _MODE_PRECISION}[target]
    except:
        assert isinstance(target, int)
        base_mode = target
    if fill_max: base_mode += _FLAG_FILLMAX
    return base_mode

cpdef double loss_overall_q_(idx2rnk, rnk2idx, labels, max_classes, ps,
                             fill_max=False,
                             a_star=0., r=0.3,
                             la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    #Can't do np.asarray as it creates memory leak on Windows (not on Ubuntu for some reason...)
    cdef int mode = _encode_mode(_MODE_NONE, fill_max=fill_max)
    cdef loss_param_struct lp = [float(a_star), float(r), NULL, NULL, int(mode), float(la), float(las), float(lc), float(lcs), int(pa), int(pas), int(pc), int(pcs)]
    cdef loss_ = loss_overall_q__(idx2rnk, rnk2idx, labels, max_classes, ps, lp)
    return loss_

cpdef double loss_class_specific_q_(idx2rnk, rnk2idx, labels, max_classes, ps,
                                    fill_max=False,
                                    a_star=0., rks=None, class_weights=None,
                                    mode=_MODE_RECALL,
                                    la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    #Checks
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    assert rks is None or idx2rnk.shape[1] == rks.shape[0], "rks issue"
    assert idx2rnk.shape[0] == labels.shape[0], "label"
    assert not isinstance(class_weights, bool), "class_weights cannot be bool now."

    #Prepare parameters
    mode = _encode_mode(mode, fill_max=fill_max)
    cdef loss_param_struct lp = [float(a_star), 0., NULL, NULL, int(mode), float(la), float(las), float(lc), float(lcs), int(pa), int(pas), int(pc), int(pcs)]

    #clean classes
    cdef double[::1] weights_
    if class_weights is not None:
        weights_ = class_weights
        lp.weights = &weights_[0]

    #clean risks
    cdef double[::1] rks_
    if rks is not None:
        rks_ = rks
        lp.r_stars = &rks_[0]

    cdef double loss_ = loss_class_specific_q__(idx2rnk, rnk2idx, labels, max_classes, ps, lp)
    return loss_


def search_full_class_specific(idx2rnk, rnk2idx, labels, max_classes, ps, k,
                               fill_max=False,
                               a_star=0., rks=None, class_weights=None,
                               mode=_MODE_RECALL,
                               la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    mode = _encode_mode(mode, fill_max=fill_max)
    assert class_weights is None, "search_full_class_specific only supports no class_weights"
    cdef loss_param_struct lp = [float(a_star), 0., NULL, NULL, int(mode), float(la), float(las), float(lc), float(lcs), int(pa), int(pas), int(pc), int(pcs)]
    #clean risks
    cdef double[::1] rks_
    if rks is not None:
        rks_ = rks
        lp.r_stars = &rks_[0]

    return search_full_class_specific_complete__(idx2rnk, rnk2idx, labels, max_classes, ps,
                                                 k, 0, -1, lp)


def search_full_overall(idx2rnk, rnk2idx, labels, max_classes, ps, k,
                        fill_max=False,
                        a_star=0., r=0.,
                        la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    cdef int mode = _encode_mode(_MODE_NONE, fill_max=fill_max)
    cdef loss_param_struct lp = [float(a_star), float(r), NULL, NULL, int(mode), float(la), float(las), float(lc), float(lcs), int(pa), int(pas), int(pc), int(pcs)]
    return search_full_overall__(idx2rnk, rnk2idx, labels, max_classes, ps, k, 0, -1, lp)


cpdef main_coord_descent_class_specific_(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                                         max_step=None, fill_max=False,
                                         a_star=0., rks=None, class_weights=None,
                                         mode=_MODE_RECALL,
                                         la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    #Checks
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    assert rks is None or idx2rnk.shape[1] == rks.shape[0]
    assert N == rnk2idx.shape[0] == labels.shape[0] == max_classes.shape[0]
    assert len(init_ps.shape) == 1, "init_ps"
    assert not isinstance(class_weights, bool), "class_weights cannot be bool now."
    cdef int[::1] new_ps = np.asarray(init_ps.copy(), np.int32)
    #print(init_ps)

    #Prepare parameters
    mode = _encode_mode(mode, fill_max=fill_max)
    cdef loss_param_struct lp = [float(a_star), 0., NULL, NULL, int(mode), float(la), float(las), float(lc), float(lcs), int(pa), int(pas), int(pc), int(pcs)]

    #clean classes
    cdef double[::1] weights_
    if class_weights is not None:
        weights_ = class_weights
        lp.weights = &weights_[0]

    #clean risks
    cdef double[::1] rks_
    if rks is not None:
        rks_ = rks
        lp.r_stars = &rks_[0]

    cdef double best_loss
    cdef int n_searches
    n_searches , best_loss = main_coord_descent_class_specific__(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                                                                 &new_ps[0], &best_loss, max_step or -1,
                                                                 lp)
    return best_loss, np.asarray(new_ps, dtype=np.int32), n_searches



cpdef main_coord_descent_overall_(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                                  max_step=None, fill_max=False,
                                  a_star=0., r=0.,
                                  la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    #Checks
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    assert N == rnk2idx.shape[0] == labels.shape[0]
    assert K == rnk2idx.shape[1] == init_ps.shape[0]
    cdef int[::1] new_ps = np.asarray(init_ps.copy(), np.int32)

    cdef int mode = _encode_mode(_MODE_NONE, fill_max=fill_max)
    cdef loss_param_struct lp = [float(a_star), float(r), NULL, NULL, int(mode), float(la), float(las), float(lc), float(lcs), int(pa), int(pas), int(pc), int(pcs)]
    cdef double best_loss
    cdef int n_searches
    n_searches , best_loss = main_coord_descent_overall__(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                                                          &new_ps[0], &best_loss, max_step or -1,
                                                          lp)
    return best_loss, np.asarray(new_ps, dtype=np.int), n_searches



#=======================================================================New Global threshold thing
cpdef main_coord_descent_class_specific_globalt_(rnk2ik, labels, max_classes, ascending=False, fill_max=False,
                                                 a_star = 0, rks=None, class_weights=None,
                                                 mode=_MODE_RECALL,
                                                 la=_D_LA, las=_D_LAS, lc=_D_LC, lcs=_D_LCS, pa=_D_PA, pas=_D_PAS, pc=_D_PC, pcs=_D_PCS):
    #Checks
    cdef int N = len(labels)
    assert len(rnk2ik) % N == 0, "rnk2ik should have all N*K ranks"
    cdef int K = len(rnk2ik) / N
    assert N * K == len(rnk2ik), "rnk2ik len issues"
    assert rnk2ik[:, 1].max() < K and rnk2ik[:, 0].max() < N, "rnk2ik val issues"
    assert not isinstance(class_weights, bool), "class_weights cannot be bool now."

    #Prepare parameters
    mode = _encode_mode(mode, fill_max=fill_max)
    cdef loss_param_struct lp = [float(a_star), 0., NULL, NULL, int(mode), float(la), float(las), float(lc), float(lcs), int(pa), int(pas), int(pc), int(pcs)]

    #clean classes
    cdef double[::1] weights_
    if class_weights is not None:
        weights_ = class_weights
        lp.weights = &weights_[0]

    #clean risks
    cdef double[::1] rks_
    if rks is not None:
        rks_ = rks
        lp.r_stars = &rks_[0]

    best_rnk, best_loss = search_full_class_specific_complete_globalt__(rnk2ik, labels, max_classes,
                                                                        1 if ascending else 0,
                                                                        lp)
    return best_rnk, best_loss