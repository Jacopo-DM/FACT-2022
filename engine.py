from socket import INADDR_ALLHOSTS_GROUP
import torch
import PIL
import numpy as np
from torch import nn
import tools.calculate_tool as cal
from tqdm.auto import tqdm
from eval import CausalMetric, auc, gkern
from infid_sen_utils import evaluate_infid_sen,get_exp_sens
def train_one_epoch(model, data_loader, optimizer, device, record, epoch,testing=False):
    model.train()
    calculation(model, "train", data_loader, device, record, epoch, optimizer,testing)


@torch.no_grad()
def evaluate(model, data_loader, device, record, epoch, testing=False):
    model.eval()
    calculation(model, "val", data_loader, device, record, epoch, testing)


def calculation(model, mode, data_loader, device, record, epoch, optimizer=None, testing=False):
    L = len(data_loader)
    running_loss = 0.0
    running_corrects = 0.0
    running_att_loss = 0.0
    running_log_loss = 0.0
    print("start " + mode + " :" + str(epoch))
    scores = {'del': [], 'ins': []}
    sensit=0.0
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)

        if mode == "train":
            optimizer.zero_grad()
        if testing:
            logits, loss_list, expl = model(inputs, labels)
            right_expl = expl[torch.arange(expl.size(0)), labels]
            batch_right_expl=right_expl.cpu().detach().numpy()
            heatmap_right=np.zeros((inputs.shape[0],260,260))
            for i in range(len(batch_right_expl)):
                slot_im= PIL.Image.fromarray(np.uint8(batch_right_expl[i]*255))
                right_size_slot=np.array(np.uint8(slot_im.resize((260,260), resample=PIL.Image.BILINEAR)))
                heatmap_right[i]=right_size_slot
            norm = np.linalg.norm(heatmap_right)
            sensit +=get_exp_sens(inputs, model,heatmap_right, labels,0.1,50,norm)
            klen = 11
            ksig = 5    
            kern = gkern(klen, ksig).to(device)
            blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

            insertion = CausalMetric(model, 'ins', 260 * 8, substrate_fn=blur)
            deletion = CausalMetric(model, 'del', 260 * 8, substrate_fn=torch.zeros_like)
            h = insertion.evaluate(inputs.squeeze(), heatmap_right,inputs.shape[0])
            scores['ins'].append(auc(h.mean(1)))
            h = deletion.evaluate(inputs.squeeze(), heatmap_right,inputs.shape[0])
            scores['del'].append(auc(h.mean(1)))
            print(scores)
        else:
            logits, loss_list,_ = model(inputs, labels)
        
        loss = loss_list[0]
        if mode == "train":
            loss.backward()
            # clip_gradient(optimizer, 1.1)
            optimizer.step()

        a = loss.item()
        running_loss += a
        if len(loss_list) > 2: # For slot training only
            running_att_loss += loss_list[2].item()
            running_log_loss += loss_list[1].item()
        running_corrects += cal.evaluateTop1(logits, labels)
        # if i_batch % 10 == 0:
        #     print("epoch: {} {}/{} Loss: {:.4f}".format(epoch, i_batch, L-1, a))
    sensit = round(sensit/L,3)
    epoch_loss = round(running_loss/L, 3)
    epoch_loss_log = round(running_log_loss/L, 3)
    epoch_loss_att = round(running_att_loss/L, 3)
    epoch_acc = round(running_corrects/L, 3)
    epoch_IAUC = np.mean(scores['ins'])
    epoch_DAUC = np.mean(scores['del'])
    print(epoch_IAUC)
    print(epoch_DAUC)

    record[mode]["loss"].append(epoch_loss)
    record[mode]["acc"].append(epoch_acc)
    record[mode]["log_loss"].append(epoch_loss_log)
    record[mode]["att_loss"].append(epoch_loss_att)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

