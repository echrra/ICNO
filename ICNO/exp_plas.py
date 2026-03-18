import os
import argparse
import matplotlib.pyplot as plt
import time
parser = argparse.ArgumentParser('Training ICNO')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='ICNO_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument("--gpu", type=str, default='3', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gate', type=float, default=0)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='ICNO_plas')
parser.add_argument('--data_path', type=str, default='/DATA/plasticity/plas_N987_T20.mat')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import scipy.io as scio
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer

loss_log_file = "./checkpoints/{}_loss.txt".format(args.save_name)
os.makedirs(os.path.dirname(loss_log_file), exist_ok=True)  

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def random_collate_fn(batch):
    shuffled_batch = []
    shuffled_u = None
    shuffled_t = None
    shuffled_a = None
    shuffled_pos = None
    for item in batch:
        pos = item[0]
        t = item[1]
        a = item[2]
        u = item[3]

        num_timesteps = t.size(0)
        permuted_indices = torch.randperm(num_timesteps)

        t = t[permuted_indices]
        u = u[..., permuted_indices]

        if shuffled_t is None:
            shuffled_pos = pos.unsqueeze(0)
            shuffled_t = t.unsqueeze(0)
            shuffled_u = u.unsqueeze(0)
            shuffled_a = a.unsqueeze(0)
        else:
            shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
            shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
            shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
            shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)

    shuffled_batch.append(shuffled_pos)
    shuffled_batch.append(shuffled_t)
    shuffled_batch.append(shuffled_a)
    shuffled_batch.append(shuffled_u)

    return shuffled_batch


def main():
    DATA_PATH = args.data_path

    N = 987
    ntrain = 900
    ntest = 80

    s1 = 101
    s2 = 31
    T = 20
    Deformation = 4

    r1 = 1
    r2 = 1
    s1 = int(((s1 - 1) / r1) + 1)
    s2 = int(((s2 - 1) / r2) + 1)

    data = scio.loadmat(DATA_PATH)
    input = torch.tensor(data['input'], dtype=torch.float)
    output = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)
    print(input.shape, output.shape)
    x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2)
    x_train = x_train.reshape(ntrain, -1, 1)
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = y_train.reshape(ntrain, -1, Deformation, T)
    x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2)
    x_test = x_test.reshape(ntest, -1, 1)
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = y_test.reshape(ntest, -1, Deformation, T)
    print(x_train.shape, y_train.shape)

    x_normalizer = UnitTransformer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    x_normalizer.cuda()

    x = np.linspace(0, 1, s1)
    y = np.linspace(0, 1, s2)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print("Dataloading is over.")

    t = np.linspace(0, 1, T)
    t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
    t_train = t.repeat(ntrain, 1)
    t_test = t.repeat(ntest, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True, collate_fn=random_collate_fn)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)

    print("Dataloading is over.")
    model = get_model(args).Model(space_dim=2,
                                  n_hidden=args.n_hidden,
                                  n_layers=args.n_layers,
                                  Time_Input=True,
                                  fun_dim=1,
                                  out_dim=Deformation,
                                  mlp_ratio=args.mlp_ratio,
                                  gate = args.gate,
                                  H=s1,
                                  W=s2,).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)
    with open(loss_log_file, "a") as f:  
        f.write("parameter:{}\n".format(count_parameters(model))) 

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')
        test_l2_step = 0
        test_l2_full = 0
        showcase = 10
        id = 0
        with torch.no_grad():
            for x, tim, fx, yy in test_loader:
                id += 1
                loss = 0
                x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
                bsz = x.shape[0]

                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    im = model(x, fx, T=input_T)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im.unsqueeze(-1)
                    else:
                        pred = torch.cat((pred, im.unsqueeze(-1)), -1)

               

                # if id < 2:
                #     plt.rcParams["font.family"] = "Serif"
                #     plt.rcParams['font.size'] = 14

                #     figure1 = plt.figure(figsize=(18, 14))
                #     figure1.text(0.08, 0.17, '\n Error', rotation=90, color='purple', fontsize=20)
                #     figure1.text(0.08, 0.34, '\n Prediction', rotation=90, color='green', fontsize=20)
                #     figure1.text(0.08, 0.52, '\n Ground Truth', rotation=90, color='red', fontsize=20)
                #     figure1.text(0.08, 0.774, '\n Input', rotation=90, color='b', fontsize=20)
                #     plt.subplots_adjust(wspace=0.2)

                #     index = 0
                #     for ids in range(4):  
                #         truth = y[ids].reshape(101, 31, 4).squeeze().detach().cpu().numpy()
                #         pred_vis = im[ids].reshape(101, 31, 4).squeeze().detach().cpu().numpy()
                #         truth_du = np.linalg.norm(truth[:, :, 2:], axis=-1)
                #         pred_du = np.linalg.norm(pred_vis[:, :, 2:], axis=-1)
                #         error_du = np.abs(pred_du - truth_du)

                
                #         X = truth[:, :, 0]
                #         Y = truth[:, :, 1]

                
                #         input_img = fx[ids, :, 0].reshape(101, 31).detach().cpu().numpy()

                #         # INPUT
                #         plt.subplot(4, 4, index + 1)
                #         plt.imshow(input_img, cmap='turbo', interpolation='Gaussian')
                #         plt.title(f'Case-{ids+1}', color='b', fontsize=18, fontweight='bold')
                #         plt.axis('off')

                #         # GROUND TRUTH (scatter)
                #         plt.subplot(4, 4, index + 1 + 4)
                #         plt.pcolormesh(X, Y,  truth_du, cmap='jet')
                #         plt.axis('off')
                #         plt.colorbar(fraction=0.045)
                #         # plt.clim(0, 6)

                #         # PREDICTION
                #         plt.subplot(4, 4, index + 1 + 8)
                #         plt.pcolormesh(X, Y,  pred_du, cmap='jet')
                #         plt.axis('off')
                #         plt.colorbar(fraction=0.045)
                #         # plt.clim(0, 6)

                #         # ERROR
                #         plt.subplot(4, 4, index + 1 + 12)
                #         plt.pcolormesh(X, Y,  error_du, cmap='jet')
                #         plt.axis('off')
                #         plt.colorbar(fraction=0.045)
                #         # plt.clim(-0.2, 0.2)

                #         index += 1

                #     plt.savefig(os.path.join('./results/' + save_name + '/', 'plas_summary.pdf'), bbox_inches='tight')
                #     plt.close()

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        print("test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(test_l2_step / ntest / T, test_l2_full / ntest))
    else:
        for ep in range(args.epochs):
            epoch_start_time = time.time()  


            model.train()
            train_l2_step = 0
            train_l2_full = 0
            
            for x, tim, fx, yy in train_loader:
                x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
                bsz = x.shape[0]
                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    im = model(x, fx, T=input_T)
                    loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    train_l2_step += loss.item()
                    if t == 0:
                        pred = im.unsqueeze(-1)
                    else:
                        pred = torch.cat((pred, im.unsqueeze(-1)), -1)

                    optimizer.zero_grad()
                    loss.backward()
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                
                train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                scheduler.step()
        

            model.eval()
            test_l2_step = 0
            test_l2_full = 0
            with torch.no_grad():
                for x, tim, fx, yy in test_loader:
                    loss = 0
                    x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
                    bsz = x.shape[0]

                    for t in range(T):
                        y = yy[..., t:t + 1]
                        input_T = tim[:, t:t + 1].reshape(bsz, 1)
                        im = model(x, fx, T=input_T)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im.unsqueeze(-1)
                        else:
                            pred = torch.cat((pred, im.unsqueeze(-1)), -1)

                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

            epoch_time = time.time() - epoch_start_time 
            print("Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} ,test_step_loss:{:.5f} , test_full_loss:{:.5f} , training time: {:.5f} seconds ".format(ep,
                                                                                                             train_l2_step / ntrain / T,
                                                                                                             train_l2_full / ntest,
                                                                                                             test_l2_step / ntest / T,
                                                                                                             test_l2_full / ntest,
                                                                                                             epoch_time))
            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))
            with open(loss_log_file, "a") as f:  
                f.write("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(ep+1,  train_l2_step / ntrain / T,train_l2_full / ntest,test_l2_step / ntest / T,test_l2_full / ntest,epoch_time))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))


if __name__ == "__main__":
    main()
