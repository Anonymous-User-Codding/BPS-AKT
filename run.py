# Code reused from https://github.com/arghosh/AKT
import numpy as np
import torch
import math
from sklearn import metrics
from utils import model_id_type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transpose_data_model = {'akt'}


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_acc(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, optimizer, s_data, sa_data, eid_data, tid_data, fid_data, xid_data, yid_data, label):
    net.train()
    eid_flag, xid_flag, yid_flag, model_type = model_id_type(params.model)
    N = int(math.ceil(len(s_data) / params.batch_size))
    s_data = s_data.T
    sa_data = sa_data.T
    # Shuffle the data
    shuffled_ind = np.arange(s_data.shape[1])
    np.random.shuffle(shuffled_ind)
    s_data = s_data[:, shuffled_ind]
    sa_data = sa_data[:, shuffled_ind]

    if eid_flag:
        eid_data = eid_data.T
        eid_data = eid_data[:, shuffled_ind]
        tid_data = tid_data.T
        tid_data = tid_data[:, shuffled_ind]
        fid_data = fid_data.T
        fid_data = fid_data[:, shuffled_ind]

    if xid_flag:
        xid_data = xid_data.T
        xid_data = xid_data[:, shuffled_ind]
    if yid_flag:
        yid_data = yid_data.T
        yid_data = yid_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    element_count = 0
    true_el = 0
    for idx in range(N):
        optimizer.zero_grad()

        s_one_seq = s_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        if eid_flag:
            eid_one_seq = eid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
            tid_one_seq = tid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
            fid_one_seq = fid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        if xid_flag:
            xid_one_seq = xid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        if yid_flag:
            yid_one_seq = yid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]

        sa_one_seq = sa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]

        if model_type in transpose_data_model:
            input_s = np.transpose(s_one_seq[:, :])  # Shape (bs, seqlen)
            input_sa = np.transpose(sa_one_seq[:, :])  # Shape (bs, seqlen)
            target = np.transpose(sa_one_seq[:, :])
            if eid_flag:
                # Shape (seqlen, batch_size)
                input_eid = np.transpose(eid_one_seq[:, :])
                input_tid = np.transpose(tid_one_seq[:, :])
                input_fid = np.transpose(fid_one_seq[:, :])
            if xid_flag:
                # Shape (seqlen, batch_size)
                input_xid = np.transpose(xid_one_seq[:, :])
            if yid_flag:
                # Shape (seqlen, batch_size)
                input_yid = np.transpose(yid_one_seq[:, :])
        else:
            input_s = (s_one_seq[:, :])  # Shape (seqlen, batch_size)
            input_sa = (sa_one_seq[:, :])  # Shape (seqlen, batch_size)
            target = (sa_one_seq[:, :])
            if eid_flag:
                input_eid = (eid_one_seq[:, :])  # Shape (seqlen, batch_size)
                input_tid = (tid_one_seq[:, :])  # Shape (seqlen, batch_size)
                input_fid = (fid_one_seq[:, :])  # Shape (seqlen, batch_size)
            if xid_flag:
                input_xid = (xid_one_seq[:, :])  # Shape (seqlen, batch_size)
            if yid_flag:
                input_yid = (yid_one_seq[:, :])  # Shape (seqlen, batch_size)
        target = (target - 1) / params.n_skill
        target_1 = np.floor(target)
        el = np.sum(target_1 >= -.9)
        element_count += el

        input_s = torch.from_numpy(input_s).long().to(device)
        input_sa = torch.from_numpy(input_sa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if eid_flag:
            input_eid = torch.from_numpy(input_eid).long().to(device)
            input_tid = torch.from_numpy(input_tid).long().to(device)
            input_fid = torch.from_numpy(input_fid).long().to(device)
        if xid_flag:
            input_xid = torch.from_numpy(input_xid).long().to(device)
        if yid_flag:
            input_yid = torch.from_numpy(input_yid).long().to(device)

        if eid_flag:
            if xid_flag:
                if yid_flag:
                    loss, pred, true_st = net(input_s, input_sa, target, input_eid, input_tid, input_fid, input_xid, input_yid)
                else:
                    loss, pred, true_st = net(input_s, input_sa, target, input_eid, input_tid, input_fid, input_xid)
            else:
                loss, pred, true_st = net(input_s, input_sa, target, input_tid, input_fid, input_eid)
        else:
            loss, pred, true_st = net(input_s, input_sa, target)
        pred = pred.detach().cpu().numpy()  # (seqlen * batch_size, 1)
        loss.backward()
        true_el += true_st.cpu().numpy()

        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=params.maxgradnorm)

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_acc(all_target, all_pred)

    return loss, acc, auc


def test(net, params, optimizer, s_data, sa_data, eid_data, tid_data, fid_data, xid_data, yid_data, label):
    # dataArray: [ array([[],[],..])]
    eid_flag, xid_flag, yid_flag, model_type = model_id_type(params.model)
    net.eval()
    N = int(math.ceil(float(len(s_data)) / float(params.batch_size)))
    s_data = s_data.T
    sa_data = sa_data.T
    if eid_flag:
        eid_data = eid_data.T
        tid_data = tid_data.T
        fid_data = fid_data.T
    if xid_flag:
        xid_data = xid_data.T
    if yid_flag:
        yid_data = yid_data.T

    seq_num = s_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0
    element_count = 0
    for idx in range(N):
        s_one_seq = s_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        if eid_flag:
            eid_one_seq = eid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
            tid_one_seq = tid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
            fid_one_seq = fid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        if xid_flag:
            xid_one_seq = xid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        if yid_flag:
            yid_one_seq = yid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]

        input_s = s_one_seq[:, :]  # Shape (seqlen, batch_size)
        sa_one_seq = sa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        input_sa = sa_one_seq[:, :]  # Shape (seqlen, batch_size)

        # print 'seq_num', seq_num
        if model_type in transpose_data_model:
            # Shape (seqlen, batch_size)
            input_s = np.transpose(s_one_seq[:, :])
            # Shape (seqlen, batch_size)
            input_sa = np.transpose(sa_one_seq[:, :])
            target = np.transpose(sa_one_seq[:, :])
            if eid_flag:
                input_eid = np.transpose(eid_one_seq[:, :])
                input_tid = np.transpose(tid_one_seq[:, :])
                input_fid = np.transpose(fid_one_seq[:, :])
            if xid_flag:
                input_xid = np.transpose(xid_one_seq[:, :])
            if yid_flag:
                input_yid = np.transpose(yid_one_seq[:, :])
        else:
            input_s = (s_one_seq[:, :])  # Shape (seqlen, batch_size)
            input_sa = (sa_one_seq[:, :])  # Shape (seqlen, batch_size)
            target = (sa_one_seq[:, :])
            if eid_flag:
                input_eid = (eid_one_seq[:, :])
                input_tid = (tid_one_seq[:, :])
                input_fid = (fid_one_seq[:, :])
            if xid_flag:
                input_xid = (xid_one_seq[:, :])
            if yid_flag:
                input_yid = (yid_one_seq[:, :])
        target = (target - 1) / params.n_skill
        target_1 = np.floor(target)
        #target = np.random.randint(0,2, size = (target.shape[0],target.shape[1]))

        input_s = torch.from_numpy(input_s).long().to(device)
        input_sa = torch.from_numpy(input_sa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if eid_flag:
            input_eid = torch.from_numpy(input_eid).long().to(device)
            input_tid = torch.from_numpy(input_tid).long().to(device)
            input_fid = torch.from_numpy(input_fid).long().to(device)
        if xid_flag:
            input_xid = torch.from_numpy(input_xid).long().to(device)
        if yid_flag:
            input_yid = torch.from_numpy(input_yid).long().to(device)

        with torch.no_grad():
            if eid_flag:
                if xid_flag:
                    if yid_flag:
                        loss, pred, st = net(input_s, input_sa, target, input_eid, input_tid, input_fid,
                                             input_xid, input_yid)
                    else:
                        loss, pred, st = net(input_s, input_sa, target, input_eid, input_tid, input_fid, input_xid)
                else:
                    loss, pred, st = net(input_s, input_sa, target, input_eid, input_tid, input_fid)
            else:
                loss, pred, st = net(input_s, input_sa, target)
        pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
        true_el += st.cpu().numpy()

        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        # print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_acc(all_target, all_pred)

    return loss, accuracy, auc
