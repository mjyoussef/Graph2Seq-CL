import torch
from torch_geometric.loader import DataLoader

import torch.optim as optim
from torchvision import transforms

import numpy as np
import pandas as pd
import os

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import sys
sys.path.append('../..')

from utils import ASTNodeEncoder, get_vocab_mapping, augment_edge, encode_y_to_arr, decode_arr_to_seq

# model
from model import Model

import datetime

def train(model, device, loader, optimizer, scheduler, multicls_criterion, epoch, alpha=0.05, 
        cl=False, cl_all=False, dgi_task=False):

    loss_accum = 0
    chkpt_folder = 'checkpoints/epoch' + str(epoch)
    if (not os.path.exists(chkpt_folder)):
        os.mkdir(chkpt_folder)

    for step, batch in enumerate(loader):

        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            labels = [batch.y[i] for i in range(len(batch.y))]
            pred_list, cl_loss, dgi_loss = model(batch, labels, training=True, cl=cl, cl_all=cl_all, dgi_task=dgi_task)
            optimizer.zero_grad()

            loss = 0
            for i in range(len(pred_list)):
                loss += (1-alpha) * multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])

            loss /= len(pred_list)

            if (cl and dgi_task):
                raise Exception("Cannot use both a contrastive and dgi loss term\n")
        
            if (cl):
                loss -= alpha * cl_loss

            if (dgi_task):
                loss -= alpha * dgi_loss

            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            optimizer.step()

            loss_accum += loss.item()
            print('Average loss after batch ' + str(step) + ': ' + str(loss_accum / (step + 1)))
        
        if ((step+1) % 35 == 0 or step == len(loader)-1): # save model after every 35 batches
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_accum / (step + 1),
            }, chkpt_folder + '/model' + str((step+1) // 35) + '.pt')

    print('Average training loss: {}'.format(loss_accum / (step + 1)))
    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator, arr_to_seq):

    seq_ref_list = []
    seq_pred_list = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                labels = [batch.y[i] for i in range(len(batch.y))]
                pred_list, _, _ = model(batch, labels) # no cl by default

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [arr_to_seq(arr) for arr in mat]

            seq_ref = labels

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
    return evaluator.eval(input_dict)

def randomly_mask(dataset, size):
    bool_mask = np.zeros(len(dataset), dtype=bool)
    bool_mask[:size] = True
    np.random.shuffle(bool_mask)
    out = dataset[bool_mask]
    return out


def main(starting_chkpt=None, cl=False, cl_all=False, dgi_task=False):
    # constants
    dataset_name = "ogbg-code2"

    num_vocab = 5000
    max_seq_len = 5

    depth = 3
    batch_size = 50
    epochs = 50
    learning_rate = 0.001
    step_size = 10
    decay_rate = 0.1
    weight_decay = 0.00005

    dim_h = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PygGraphPropPredDataset(dataset_name)

    split_idx = dataset.get_idx_split()

    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)

    dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

    evaluator = Evaluator(dataset_name)

    full_training = randomly_mask(dataset[split_idx["train"]], batch_size*400)
    full_valid = randomly_mask(dataset[split_idx["valid"]], batch_size*400)
    full_test = randomly_mask(dataset[split_idx["test"]], batch_size*400)

    train_loader = DataLoader(full_training, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(full_valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    node_encoder = ASTNodeEncoder(dim_h, num_nodetypes=len(nodetypes_mapping['type']), num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)

    model = Model(batch_size, depth, dim_h, max_seq_len, node_encoder, vocab2idx, device).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_rate)

    multicls_criterion = torch.nn.CrossEntropyLoss()

    starting_epoch = 1

    if (starting_chkpt != None):
        checkpoint = torch.load(starting_chkpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        starting_epoch = checkpoint['epoch']


    valid_curve = []
    test_curve = []
    train_curve = []
    trainL_curve = []

    for epoch in range(starting_epoch, epochs + 1):
        print (datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        print("Epoch {} training...".format(epoch))
        print ("lr: ", optimizer.param_groups[0]['lr'])
        train_loss = train(model, device, train_loader, optimizer, scheduler, multicls_criterion, epoch, cl=cl, cl_all=cl_all, dgi_task=dgi_task)

        scheduler.step()

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        valid_perf = eval(model, device, valid_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        test_perf = eval(model, device, test_loader, evaluator, arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))

        print('Train:', train_perf[dataset.eval_metric],
              'Validation:', valid_perf[dataset.eval_metric],
              'Test:', test_perf[dataset.eval_metric],
              'Train loss:', train_loss)

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        trainL_curve.append(train_loss)

    print('F1')
    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    print('Finished test: {}, Validation: {}, Train: {}, epoch: {}, best train: {}, best loss: {}'
          .format(test_curve[best_val_epoch], valid_curve[best_val_epoch], train_curve[best_val_epoch],
                  best_val_epoch, best_train, min(trainL_curve)))

if __name__ == "__main__":
    main(cl=False, cl_all=False, dgi_task=False)