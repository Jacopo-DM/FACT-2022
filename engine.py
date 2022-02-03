from socket import INADDR_ALLHOSTS_GROUP
import torch
import PIL
import numpy as np
from torch import nn
import tools.calculate_tool as cal
from tqdm.auto import tqdm
from IAUC_DAUC_eval import CausalMetric, auc, gkern
from infid_sen_utils import get_exp_sens_neg, get_exp_sens
import os
import os.path
from bs4 import BeautifulSoup

def train_one_epoch(model, data_loader, optimizer, device, record, epoch,testing=False):
    model.train()
    calculation(model, "train", data_loader, device, record, epoch, optimizer,testing)


@torch.no_grad()

def evaluate(model, data_loader, device, record, epoch, testing=False,anno_dir=None, loss_status=None):
    model.eval()
    if testing:
        res_dic=calculation(model, "val", data_loader, device, record, epoch, testing=testing,anno_dir=anno_dir,loss_status=loss_status)
        return res_dic
    else:
        calculation(model, "val", data_loader, device, record, epoch, testing=testing,anno_dir=anno_dir,loss_status=loss_status)


def precision(heatmap, borders,xmax,ymax):

    xmax = xmax.cpu().detach().numpy()
    ymax = ymax.cpu().detach().numpy()
    x_ratio=260/xmax
    y_ratio=260/ymax
    total=0
    for i in heatmap:
        for j in i:
            total+=j
    hits=0

    for border in borders:
        for i in range(round(border[2]*y_ratio),round(border[3]*y_ratio)-1):
            for j in range(round(border[0]*x_ratio),round(border[1]*x_ratio)-1):
                hits+=heatmap[i][j]
    return hits/total



def calculation(model, mode, data_loader, device, record, epoch, optimizer=None, testing=False, anno_dir=None,loss_status=None):
    L = len(data_loader)
    running_loss = 0.0
    running_corrects = 0.0
    running_att_loss = 0.0
    running_log_loss = 0.0
    print("start " + mode + " :" + str(epoch))
    scores = {'del': [], 'ins': []}
    sensit=0.0
    prec=0.0
    area_size=0.0
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        names= sample_batch["names"]
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        widths = sample_batch["width"]
        heights =sample_batch["height"]




        if mode == "train":
            optimizer.zero_grad()
        if testing:
            borders=[]
            batch_xmax=0
            batch_ymax=0

            for name in names:
                replaced=name.replace('_', '.')
                x = replaced.split(".")
                for i in range(len(x)):
                    if x[i-1]=='val':
                        pic_id=x[i]
                a=os.path.join(anno_dir+"val/ILSVRC2012_val_"+pic_id+'.xml')
                with open(a) as f:
                    data = f.read()
                Bs_data = BeautifulSoup(data, "xml")
                xmin = Bs_data.find_all('xmin')
                xmax = Bs_data.find_all('xmax')
                ymin = Bs_data.find_all('ymin')
                ymax = Bs_data.find_all('ymax')
                brd=[]
                for i in range(len(xmin)):
                    if int(xmax[i].string)>batch_xmax:
                        batch_xmax=int(xmax[i].string)

                    if int(ymax[i].string)>batch_ymax:
                        batch_ymax=int(ymax[i].string)
                    brd.append([int(xmin[i].string),int(xmax[i].string),int(ymin[i].string),int(ymax[i].string)])
                borders.append(brd)
            borders=np.array(borders, dtype="object")
            logits, loss_list, expl = model(inputs, labels)
            ground_truth= torch.argmax(logits, dim=1)
            LSC=torch.argmin(logits, dim=1)
            right_label=ground_truth.view(-1, 1, 1).expand(inputs.shape[0], 1, 81)
            wrong_label=LSC.view(-1, 1, 1).expand(inputs.shape[0], 1, 81)

            batch_right_expl=torch.gather(expl, 1, right_label)
            batch_right_expl=batch_right_expl.squeeze()
            batch_right_expl=batch_right_expl.reshape(expl.size(0),9,9).cpu().detach().numpy()

            batch_wrong_expl=torch.gather(expl, 1, wrong_label)
            batch_wrong_expl=batch_wrong_expl.squeeze()
            batch_wrong_expl=batch_wrong_expl.reshape(expl.size(0),9,9).cpu().detach().numpy()
            heatmap_right=np.zeros((inputs.shape[0],260,260))
            heatmap_wrong=np.zeros((inputs.shape[0],260,260))
            batch_prec=0
            batch_area=0

            for i in range(len(batch_right_expl)):
                slot_im_wrong= PIL.Image.fromarray(batch_wrong_expl[i])
                wrong_size_slot=np.array(slot_im_wrong.resize((260,260), resample=PIL.Image.BILINEAR))
                slot_im_right= PIL.Image.fromarray(batch_right_expl[i])
                right_size_slot=np.array(slot_im_right.resize((260,260), resample=PIL.Image.BILINEAR))
                heatmap_right[i] = right_size_slot
                heatmap_wrong[i] = wrong_size_slot
                if loss_status==-1:
                    batch_prec +=precision(wrong_size_slot,borders[i],widths[i],heights[i])
                else:
                    batch_prec +=precision(heatmap_right[i],borders[i],widths[i],heights[i])
                slot_image_size = heatmap_right[i].shape
                batch_area += float(heatmap_right[i].sum()) / float(slot_image_size[0]*slot_image_size[1])
            
            if loss_status==-1:
                norm = np.linalg.norm(heatmap_wrong)
                sensit +=get_exp_sens_neg(inputs, model,heatmap_wrong, labels,0.1,50,norm)
            else:
                norm = np.linalg.norm(heatmap_right)
                sensit +=get_exp_sens(inputs, model,heatmap_right, labels,0.1,50,norm)

            prec+=batch_prec/inputs.shape[0]
            area_size+=batch_area/inputs.shape[0]
            klen = 11
            ksig = 5    
            kern = gkern(klen, ksig).to(device)
            blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

            insertion = CausalMetric(model, 'ins', 260*8, substrate_fn=blur)
            deletion = CausalMetric(model, 'del', 260*8, substrate_fn=torch.zeros_like)
            h = insertion.evaluate(inputs.squeeze(), heatmap_right,inputs.shape[0])
            scores['ins'].append(auc(h.mean(1)))
            h = deletion.evaluate(inputs.squeeze(), heatmap_right,inputs.shape[0])
            scores['del'].append(auc(h.mean(1)))
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
    total_prec = round(prec/L,3)
    total_area = round(area_size/L,3)
    sensit = round(sensit/L,3)
    epoch_loss = round(running_loss/L, 3)
    epoch_loss_log = round(running_log_loss/L, 3)
    epoch_loss_att = round(running_att_loss/L, 3)
    epoch_acc = round(running_corrects/L, 3)
    epoch_IAUC = np.mean(scores['ins'])
    epoch_DAUC = np.mean(scores['del'])
    print('IAUC: ', epoch_IAUC)
    print('DAUC: ',epoch_DAUC)
    print('sensitivity: ', sensit)
    print('area_size: ',total_area)
    print('precision: ',total_prec)
    record[mode]["loss"].append(epoch_loss)
    record[mode]["acc"].append(epoch_acc)
    record[mode]["log_loss"].append(epoch_loss_log)
    record[mode]["att_loss"].append(epoch_loss_att)
    if testing:
        return {"IAUC": epoch_IAUC, "DAUC": epoch_DAUC, "acc":epoch_acc, "precision":total_prec,"area_size":total_area,"sensitivity": sensit}


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

