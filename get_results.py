import argparse
from engine import evaluate
from train import get_args_parser, param_translation
import os
import torch
from sloter.slot_model import SlotModel
from dataset.choose_dataset import select_dataset
from tools.prepare_things import DataLoaderX
from pathlib import Path
from tools.calculate_tool import MetricLog
def main(args):
    device = torch.device(args.device)

    model = SlotModel(args)
    print("evaluate model: " + f"{'use slot ' if args.use_slot else 'without slot '}" + f"{'negetive loss' if args.use_slot and args.loss_status != 1 else 'positive loss'}")
    model.to(device)
    model_without_ddp = model
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train, dataset_val  = select_dataset(args)

    log = MetricLog()
    record = log.record
        
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoaderX(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.pre_dir, map_location=device.type)
        model_without_ddp.load_state_dict(checkpoint['model'])
    results_dic=evaluate(model_without_ddp, data_loader_val, device, record, args.epochs,testing=True,anno_dir=args.annotations_dir, loss_status=args.loss_status)
    return results_dic

def calculate_metrics(arguments):
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args(arguments)
    args_dict = vars(args)
    args_for_evaluation = ['num_classes', 'lambda_value', 'power', 'slots_per_class']
    args_type = [int, float, int, int]
    target_arg = None
    for arg_id, arg in enumerate(args_for_evaluation):
        if args_dict[arg].find(',') > 0:
            target_arg = arg
        else:
            args_dict[arg] = args_type[arg_id](args_dict[arg])

    if target_arg is None:
        results_dic=main(args)
    return results_dic
