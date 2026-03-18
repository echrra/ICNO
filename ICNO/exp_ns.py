import os
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import numpy as np
import time

parser = argparse.ArgumentParser('Training ICNO')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='ICNO_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument("--gpu", type=str, default='5', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gate', type=float, default=1)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='ICNO_ns_2d')
parser.add_argument('--data_path', type=str, default='/DATA')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import scipy.io as scio
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model

loss_log_file = "./checkpoints/{}_loss.txt".format(args.save_name)
os.makedirs(os.path.dirname(loss_log_file), exist_ok=True)  


data_path = args.data_path + '/NavierStokes_V1e-5_N1200_T20.mat'

ntrain = 1000
ntest = 200
T_in = 10
T = 10
step = 1
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
    r = args.downsample
    h = int(((64 - 1) / r) + 1)

    data = scio.loadmat(data_path)
    print(data['u'].shape)
    train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
    train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
    train_a = torch.from_numpy(train_a)
    train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)

    test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
    test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
    test_a = torch.from_numpy(test_a)
    test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                              batch_size=args.batch_size, shuffle=False)

    print("Dataloading is over.")

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=T_in,
                                  out_dim=1,
                                  gate = args.gate,
                                  H=h, W=h,).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)
    with open(loss_log_file, "a") as f:  
        f.write("参数量：{}\n".format(count_parameters(model)))    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        showcase = 10
        id = 0

        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        test_l2_full = 0
        with torch.no_grad():
            for x, fx, yy in test_loader:
                id += 1
                
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                bsz = x.shape[0]
                for t in range(0, T, step):
                    im = model(x, fx=fx)

                    fx = torch.cat((fx[..., step:], im), dim=-1)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

               
                # if id < 2 :
                #     plt.rcParams["font.family"] = "Serif"
                #     plt.rcParams['font.size'] = 14

                #     figure1 = plt.figure(figsize=(18, 14))
                #     figure1.text(0.1, 0.17, '\n Error', rotation=90, color='purple', fontsize=20)
                #     figure1.text(0.1, 0.34, '\n Prediction', rotation=90, color='green', fontsize=20)
                #     figure1.text(0.1, 0.52, '\n Ground Truth', rotation=90, color='red', fontsize=20)
                #     figure1.text(0.1, 0.774, '\n Input', rotation=90, color='b', fontsize=20)
                #     plt.subplots_adjust(wspace=0.3, hspace=0.4)

                #     index = 0
                #     # print(id)
                #     for ids in range(4):
                #         print(ids)
                
                #         preds = im[ids, :, 0].reshape(64, 64).detach().cpu().numpy()
                #         input = test_a[ids,:,0].reshape(64, 64).detach().cpu().numpy()
                
                #         gt = yy[ids, :, t].reshape(64, 64).detach().cpu().numpy()

                #         # 
                #         error = np.abs(preds - gt)

                #         # INPUT 
                #         plt.subplot(4, 4, index + 1)
                #         plt.imshow(input, cmap='jet', interpolation='Gaussian')
                #         plt.title(f'Case-{ids+1}', color='b', fontsize=18, fontweight='bold')
                #         plt.axis('off')
                        

                #         # GROUND TRUTH
                #         plt.subplot(4, 4, index + 1 + 4)
                #         plt.imshow(gt, cmap='jet', interpolation='Gaussian')
                #         plt.axis('off')
                #         plt.colorbar(fraction=0.045)

                #         # PREDICTION
                #         plt.subplot(4, 4, index + 1 + 8)
                #         plt.imshow(preds, cmap='jet', interpolation='Gaussian')
                #         plt.axis('off')
                #         plt.colorbar(fraction=0.045)

                #         # ERROR
                #         plt.subplot(4, 4, index + 1 + 12)
                #         plt.imshow(error, cmap='jet', interpolation='Gaussian')
                #         plt.axis('off')
                #         plt.colorbar(fraction=0.045, format='%.0e')

                #         index += 1

                    

                #     plt.savefig(os.path.join('./results/' + save_name + '/', 'ns_summary.pdf'), bbox_inches='tight')
                #     plt.close()

                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

                
            print(test_l2_full / ntest)
    else:
        for ep in range(args.epochs):
            epoch_start_time = time.time()  

            model.train()
            train_l2_step = 0
            train_l2_full = 0

            for x, fx, yy in train_loader:
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x: B,4096,2    fx: B,4096,T   y: B,4096,T
                bsz = x.shape[0]

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(x, fx=fx)  # B , 4096 , 1
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    fx = torch.cat((fx[..., step:], y), dim=-1)  # detach() & groundtruth

                train_l2_step += loss.item()
                train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            test_l2_step = 0
            test_l2_full = 0

            model.eval()

            with torch.no_grad():
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                    bsz = x.shape[0]
                    for t in range(0, T, step):
                        y = yy[..., t:t + step]
                        im = model(x, fx=fx)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)
                        fx = torch.cat((fx[..., step:], im), dim=-1)

                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

            epoch_time = time.time() - epoch_start_time 
            print(
                "Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}, training time: {:.5f} seconds".format(
                    ep, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
                        test_l2_full / ntest,epoch_time))
            


            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))
            with open(loss_log_file, "a") as f:  
                f.write("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(ep+1,  train_l2_step / ntrain / T,train_l2_full / ntrain,test_l2_step / ntest / T,test_l2_full / ntest,epoch_time))
                
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))


if __name__ == "__main__":
    main()
