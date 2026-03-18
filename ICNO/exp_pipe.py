import os
import argparse
import matplotlib.pyplot as plt
import time
parser = argparse.ArgumentParser('Training ICNO')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='ICNO_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--n-heads', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='7', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gate', type=float, default=1)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='ICNO_pipe')
parser.add_argument('--data_path', type=str, default='/DATA/pipe')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

loss_log_file = "./checkpoints/{}_loss.txt".format(args.save_name)
os.makedirs(os.path.dirname(loss_log_file), exist_ok=True) 


import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    INPUT_X = args.data_path + '/Pipe_X.npy'
    INPUT_Y = args.data_path + '/Pipe_Y.npy'
    OUTPUT_Sigma = args.data_path + '/Pipe_Q.npy'

    ntrain = 1000
    ntest = 200
    N = 1200

    r1 = args.downsamplex
    r2 = args.downsampley
    s1 = int(((129 - 1) / r1) + 1)
    s2 = int(((129 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 0]
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)
    x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    x_train = x_train.reshape(ntrain, -1, 2)
    x_test = x_test.reshape(ntest, -1, 2)
    y_train = y_train.reshape(ntrain, -1)
    y_test = y_test.reshape(ntest, -1)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)


    x_normalizer.cuda()
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, x_train, y_train),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, y_test),
                                              batch_size=args.batch_size,
                                              shuffle=False)

    print("Dataloading is over.")

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=0,
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
        showcase = 10
        id = 0

        with torch.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

                # if id <2:

                #     plt.rcParams["font.family"] = "Serif"
                #     plt.rcParams['font.size'] = 14

                #     figure1 = plt.figure(figsize=(18, 14))
                #     figure1.text(0.08, 0.17, '\n Error', rotation=90, color='purple', fontsize=20)
                #     figure1.text(0.08, 0.34, '\n Prediction', rotation=90, color='green', fontsize=20)
                #     figure1.text(0.08, 0.52, '\n Ground Truth', rotation=90, color='red', fontsize=20)
                #     figure1.text(0.08, 0.774, '\n Input', rotation=90, color='b', fontsize=20)
                #     plt.subplots_adjust(wspace=0.7, hspace=0.4)

                #     index = 0
                #     for ids in range(4):  # 可根据需要调整展示样本数
                #         x_coord = x[ids, :, 0].reshape(129, 129).detach().cpu().numpy()
                #         y_coord = x[ids, :, 1].reshape(129, 129).detach().cpu().numpy()
                #         input_field = np.full_like(x_coord, 0.4)
                #         gt_field = y[ids, :].reshape(129, 129).detach().cpu().numpy()
                #         pred_field = out[ids, :].reshape(129, 129).detach().cpu().numpy()
                #         error_field = np.abs(pred_field - gt_field)

                #         # Input
                #         plt.subplot(4, 4, index + 1)
                #         plt.pcolormesh(x_coord, y_coord, input_field, shading='auto', edgecolors='black', linewidths=0.1,cmap='Blues',vmin=0, vmax=1)
                #         plt.title(f'Case-{ids+1}', color='b', fontsize=18, fontweight='bold')
                        
                #         plt.axis('off')

                #         # Ground Truth
                #         plt.subplot(4, 4, index + 1 + 4)
                #         plt.pcolormesh(x_coord, y_coord, gt_field, shading='auto', cmap='coolwarm')
                #         plt.colorbar(fraction=0.045)
                #         plt.clim(0, 0.3)
                #         plt.axis('off')

                #         # Prediction
                #         plt.subplot(4, 4, index + 1 + 8)
                #         plt.pcolormesh(x_coord, y_coord, pred_field, shading='auto', cmap='coolwarm')
                #         plt.colorbar(fraction=0.045)
                #         plt.clim(0, 0.3)
                #         plt.axis('off')

                #         # Error
                #         plt.subplot(4, 4, index + 1 + 12)
                #         plt.pcolormesh(x_coord, y_coord, error_field, shading='auto', cmap='jet')
                #         plt.colorbar(fraction=0.045, format='%.0e')
                #         # plt.clim(-0.02, 0.02)
                #         plt.axis('off')

                #         index += 1

                #     plt.savefig(os.path.join('./results/' + save_name + '/', 'pipe_summary.pdf'), bbox_inches='tight')
                #     plt.close()

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))
    else:
        
        for ep in range(args.epochs):
            
            epoch_start_time = time.time() 
            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()  # x:B,N,2  fx:B,N,2  y:B,N
                optimizer.zero_grad()
                out = model(x, None).squeeze(-1)

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
                    out = model(x, None).squeeze(-1)
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
