

import os
import argparse
import numpy as np
import scipy.io as scio
import time

parser = argparse.ArgumentParser('Training ICNO')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='ICNO_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim') 
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='7', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--gate', type=float, default=0)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='ICNO_era5_2t')
parser.add_argument('--data_path', type=str, default='/DATA')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


loss_log_file = "./checkpoints/{}_loss.txt".format(args.save_name)
os.makedirs(os.path.dirname(loss_log_file), exist_ok=True) 

import torch
import torch.nn.functional as F
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer
import matplotlib.pyplot as plt
import xarray as xr

data_path = args.data_path + '/ERA5/ERA5_2mT.grib'

r = 6 # 2**4 for 4^o, # 2**3 for 2^o
s1 = int(((720 - 1) / r) + 1)
s2 = int(((1440 - 1) / r) + 1)

ntrain = 800
ntest = 200

eval = args.eval
save_name = args.save_name


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():

    ds = xr.open_dataset(data_path, engine='cfgrib')
    data = np.array(ds["t2m"])
    data = torch.tensor(data)
    data = data[:, :720, :]
    print(data.shape)
    # %%
    x_train = data[:-1][:ntrain, ::r, ::r]
    y_train = data[1:][:ntrain, ::r, ::r]

    x_test = data[:-1][-ntest:, ::r, ::r]
    y_test = data[1:][-ntest:, ::r, ::r]




    x_train = x_train.reshape(ntrain, -1)#[B, 4140, 7]
    x_test = x_test.reshape(ntest, -1)
    y_train = y_train.reshape(ntrain, -1)
    y_test = y_test.reshape(ntest, -1)


    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    x = np.linspace(0, 1, s1)
    y = np.linspace(0, 1, s2)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train,x_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test,x_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Dataloading is over.")

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=1,
                                  out_dim=1,
                                  gate = args.gate,
                                  H=s1, W=s2,).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    # print(model)
    count_parameters(model)
    with open(loss_log_file, "a") as f: 
        f.write("parameter:{}\n".format(count_parameters(model))) 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '_resave' + '.pt'))
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        
        rel_err = 0.0
        id = 0
        preds, trues = [], []

        with torch.no_grad():
            for pos, fx, y in test_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl

            
                
                preds.append(out.detach().cpu())
                trues.append(y.detach().cpu())


   
            pred = torch.cat(preds, dim=0)
            true = torch.cat(trues, dim=0)

            pred = pred.reshape(pred.shape[0], s1, s2)
            true = true.reshape(true.shape[0], s1, s2)
            print("pred shape:", pred.shape,"pred min:",pred.min().item(),"pred max:",pred.max().item())
            print("true shape:", true.shape)

            num_test = true.shape[0]

            indices = [int(num_test * 0.61), int(num_test * 0.78), int(num_test * 0.95)]
            years = [2016, 2019, 2022]

            figure1 = plt.figure(figsize=(18, 11))
            plt.subplots_adjust(hspace=0.01, wspace=0.16, top=0.9, bottom=0.10, left=0.12, right=0.95)

            for col, (idx, year) in enumerate(zip(indices, years), 1):

                plt.subplot(3, 3, col)
                plt.imshow(true[idx].numpy(), cmap='plasma', extent=[0, 360, -90, 90], origin='lower',interpolation='Gaussian')
                plt.title(f'Jan-{year}',fontsize=19)
                plt.xlabel('Longitude ($^{\circ}$)', fontsize=13)
                plt.ylabel('Latitude ($^{\circ}$)', fontsize=13)
                plt.axis('on') 

                # Prediction
                plt.subplot(3, 3, col + 3)
                plt.imshow(pred[idx].numpy(), cmap='plasma', extent=[0, 360, -90, 90], origin='lower',interpolation='Gaussian')
                plt.xlabel('Longitude ($^{\circ}$)', fontsize=13)
                plt.ylabel('Latitude ($^{\circ}$)', fontsize=13)

                # Error
                plt.subplot(3, 3, col + 6)
                plt.imshow((pred[idx] - true[idx]).numpy(), cmap='coolwarm', extent=[0, 360, -90, 90], origin='lower',interpolation='Gaussian')
                # plt.colorbar(fraction=0.024, pad=0.01)
                plt.xlabel('Longitude ($^{\circ}$)', fontsize=13)
                plt.ylabel('Latitude ($^{\circ}$)', fontsize=13)

            figure1.text(0.06, 0.78, 'Ground Truth', rotation=90, color='red', fontsize=20, va='center')
            figure1.text(0.06, 0.50, 'Prediction', rotation=90, color='green', fontsize=20, va='center')
            figure1.text(0.06, 0.22, 'Error', rotation=90, color='purple', fontsize=20, va='center')


            save_fig = f'./results/{save_name}/summary_t.png'
            os.makedirs(os.path.dirname(save_fig), exist_ok=True)
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
            plt.close()
            print(f" Visualization saved to {save_fig}")
                    

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))
    else:
        for ep in range(args.epochs):
            epoch_start_time = time.time() 
            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()  # x:B,N  fx:B,N  y:B,N
                optimizer.zero_grad()
                out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)

                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                loss = myloss(out, y)
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
                scheduler.step()

            train_loss = train_loss / ntrain
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            model.eval()
            rel_err = 0.0
            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                    out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                    out = y_normalizer.decode(out)

                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            epoch_time = time.time() - epoch_start_time  
            print("rel_err:{}".format(rel_err),"training time: {:.5f} seconds".format(epoch_time))

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))
           
            with open(loss_log_file, "a") as f:  
                f.write("{}\t{}\t{}\t{:.5f}\n".format(ep+1, train_loss, rel_err, epoch_time))
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))


if __name__ == "__main__":
    main()