import torch
from torch.autograd import Variable
import copy
import numpy as np
import math
import PIL
FORWARD_BZ = 25


def forward_batch(model, input, batchsize):
    inputsize = input.shape[0]
    for count in range((inputsize - 1) // batchsize + 1):
        end = min(inputsize, (count + 1) * batchsize)
        if count == 0:
            tempinput = input[count * batchsize:end]
            out = model(tempinput.cuda())
            out = out.data.cpu().numpy()
        else:
            tempinput = input[count * batchsize:end]
            temp = model(tempinput.cuda()).data
            out = np.concatenate([out, temp.cpu().numpy()], axis=0)
    return out


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def sample_eps_Inf(image, epsilon, N):
    images = np.tile(image, (N, 1, 1, 1))
    dim = images.shape
    return np.random.uniform(-1 * epsilon, epsilon, size=dim)





def kernel_regression(Is, ks, ys):
    """
    *Inputs:
        I: sample of perturbation of interest, shape = (n_sample, n_feature)
        K: kernel weight
    *Return:
        expl: explanation minimizing the weighted least square
    """
    n_sample, n_feature = Is.shape
    IIk = np.matmul(np.matmul(Is.transpose(), np.diag(ks)), Is)
    Iyk = np.matmul(np.matmul(Is.transpose(), np.diag(ks)), ys)
    expl = np.matmul(np.linalg.pinv(IIk), Iyk)
    return expl


def set_zero_infid(array, size, point, pert):
    if pert == "Gaussian":
        ind = np.random.choice(size, point, replace=False)
        randd = np.random.normal(size=point) * 0.2 + array[ind]
        randd = np.minimum(array[ind], randd)
        randd = np.maximum(array[ind] - 1, randd)
        array[ind] -= randd
        return np.concatenate([array, ind, randd])
    elif pert == "SHAP":
        nz_ind = np.nonzero(array)[0]
        nz_ind = np.arange(array.shape[0])
        num_nz = len(nz_ind)
        bb = 0
        while bb == 0 or bb == num_nz:
            aa = np.random.rand(num_nz)
            bb = np.sum(aa < 0.5)
        sample_ind = np.where(aa < 0.5)
        array[nz_ind[sample_ind]] = 0
        ind = np.zeros(array.shape)
        ind[nz_ind[sample_ind]] = 1
        return np.concatenate([array, ind])


def shap_kernel(Z, X):
    M = X.shape[0]
    z_ = np.count_nonzero(Z)
    return (M-1) * 1.0 / (z_ * (M - 1 - z_) * nCr(M - 1, z_))




def get_exp(ind, exp):
    return (exp[ind.astype(int)])


def get_imageset(image_copy, im_size, rads=[2, 3, 4, 5, 6]):
    rangelist = np.arange(np.prod(im_size)).reshape(im_size)
    width = 5
    height = 5
    print(width)
    ind = np.zeros(image_copy.shape)
    count = 0
    for rad in rads:
        for i in range(width - rad):
            for j in range(height - rad):
                print(j+rad)
                print(i+rad)
                ind[count, rangelist[:, i:i+rad, j:j+rad].flatten()] = 1
                image_copy[count, rangelist[:, i:i+rad, j:j+rad].flatten()] = 0
                count += 1
    return image_copy, ind


def get_exp_infid(image, model, exp, label, pdt, binary_I, pert):
    point = 784
    total = (np.prod(exp.shape))
    num = 1
    if pert == 'Square':
        im_size = image.shape
        width = im_size[2]
        height = im_size[3]
        rads = np.arange(10) + 1
        num = 0
        for rad in rads:
            num += (width - rad + 1) * (height - rad + 1)
    exp = np.squeeze(exp)
    image_copy = np.tile(np.reshape(np.copy(image.cpu()), -1), [50,1])

    exp_copy = np.reshape(np.copy(exp), -1)

    image = torch.squeeze(image)
    if pert == 'Gaussian':

        image_copy_ind = np.apply_along_axis(set_zero_infid, 1, image_copy, total, point, pert)
    elif pert == 'Square':
        image_copy, ind = get_imageset(image_copy, im_size[1:], rads=rads)

    if pert == 'Gaussian' and not binary_I:
        image_copy = image_copy_ind[:, :total]
        ind = image_copy_ind[:, total:total+point]
        rand = image_copy_ind[:, total+point:total+2*point]
        exp_sum = np.sum(rand*np.apply_along_axis(get_exp, 1, ind, exp_copy), axis=1)
        ks = np.ones(num)
    elif pert == 'Square' and binary_I:
        exp_sum = np.sum(ind * np.expand_dims(exp_copy, 0), axis=1)
        ks = np.apply_along_axis(shap_kernel, 1, ind, X=image.reshape(-1))
        ks = np.ones(num)
    else:
        raise ValueError("Perturbation type and binary_I do not match.")

    image_copy = np.reshape(image_copy, (50, 3, 260, 260))
    image_v = Variable(torch.from_numpy(image_copy.astype(np.float32)).cuda(), requires_grad=False)
    out = forward_batch(model, image_v, FORWARD_BZ)
    pdt_rm = (out[:, label])
    pdt_diff = pdt - pdt_rm

    # performs optimal scaling for each explanation before calculating the infidelity score
    beta = np.mean(ks*pdt_diff*exp_sum) / np.mean(ks*exp_sum*exp_sum)
    exp_sum *= beta
    infid = np.mean(ks*np.square(pdt_diff-exp_sum)) / np.mean(ks)
    return infid

def get_exp_sens(X, model, expl, yy, sen_r,sen_N,norm):
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda()
        X_noisy = X + sample
        logits, _, expl_noise= model(X_noisy, yy)
        ground_truth= torch.argmax(logits, dim=1)
        right_label=ground_truth.view(-1, 1, 1).expand(X_noisy.shape[0], 1, 81)
        batch_right_expl=torch.gather(expl_noise, 1, right_label)
        batch_right_expl=batch_right_expl.squeeze()
        batch_right_expl=batch_right_expl.reshape(X_noisy.shape[0],9,9).cpu().detach().numpy()
        heatmap_right=np.zeros((X_noisy.shape[0],260,260))
        for i in range(len(batch_right_expl)):
            slot_im= PIL.Image.fromarray(batch_right_expl[i])
            right_size_slot=np.array(slot_im.resize((260,260), resample=PIL.Image.BILINEAR))
            heatmap_right[i]=right_size_slot
        max_diff = max(max_diff, np.linalg.norm(expl-heatmap_right)) / norm
    return max_diff

def get_exp_sens_neg(X, model, expl, yy, sen_r,sen_N,norm):
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda()
        X_noisy = X + sample
        logits, _, expl_noise= model(X_noisy, yy)
        ground_truth= torch.argmin(logits, dim=1)
        right_label=ground_truth.view(-1, 1, 1).expand(X_noisy.shape[0], 1, 81)
        batch_right_expl=torch.gather(expl_noise, 1, right_label)
        batch_right_expl=batch_right_expl.squeeze()
        batch_right_expl=batch_right_expl.reshape(X_noisy.shape[0],9,9).cpu().detach().numpy()
        heatmap_right=np.zeros((X_noisy.shape[0],260,260))
        for i in range(len(batch_right_expl)):
            slot_im= PIL.Image.fromarray(batch_right_expl[i])
            right_size_slot=np.array(slot_im.resize((260,260), resample=PIL.Image.BILINEAR))
            heatmap_right[i]=right_size_slot
        max_diff = max(max_diff, np.linalg.norm(expl-heatmap_right)) / norm
    return max_diff

def evaluate_infid_sen(loader, model, exp, predicted, pert, sen_r, sen_N, sg_r=None, sg_N=None, given_expl=None):
    if pert == 'Square':
        binary_I = True
    elif pert == 'Gaussian':
        binary_I = False
    else:
        raise NotImplementedError('Only support Square and Gaussian perturbation.')

    model.eval()
    infids = []
    max_sens = []

    for i, X in enumerate(loader):
        if i >= 5:  # i >= 50 for the experiments used in the paper
            break
        X, y = X[0].cuda(), X[1].cuda().long()
        X=torch.unsqueeze(X, 0)
        if y.dim() == 2:
            y = y.squeeze(1)
        expl = exp[i]
        pdt = predicted[i]

        pdt = pdt.data.cpu().numpy()
        norm = np.linalg.norm(expl)
        infid = get_exp_infid(X, model, expl, y[0], pdt, binary_I=binary_I, pert=pert)

        sens = get_exp_sens(X, model, expl,exp, y[0], pdt, sg_r, sg_N,sen_r,sen_N,norm,binary_I,given_expl)
        infids.append(infid)

        max_diff = -math.inf
        #for _ in range(sen_N):   
            #sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda()
            #X_noisy = X + sample
            #expl_eps, _ = get_explanation_pdt(X_noisy, model, y[0], exp, sg_r, sg_N,
            #                                  given_expl=given_expl, binary_I=binary_I)
            #max_diff = max(max_diff, np.linalg.norm(expl-expl_eps)) / norm
        max_sens.append(sens)

    infid = np.mean(infids)
    max_sen = np.mean(max_sens)

    return infid, max_sen
