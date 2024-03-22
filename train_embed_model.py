import argparse
from args import get_args
import torch
from torch.utils.data import DataLoader
from dataset import Random_seq_seq_d_dataset
from model import Embed_model, Twin_model
from tqdm import tqdm
import os

def PNLL(output, target):
    neg_logg = output - target * torch.log(output)
    mask_out = (target>=2).to(torch.float) * (output>target).to(torch.float)
    neg_logg = neg_logg * (1-mask_out)
    return torch.mean(neg_logg)

def MSE(output, target):
    mse = (output - target)**2
    mask_out = (target>=2).to(torch.float) * (output>target).to(torch.float)
    mse = mse * (1-mask_out)
    return torch.mean(mse)

def train(args):
    rand_dataset = Random_seq_seq_d_dataset(args.length, args.padded_length, seed=args.seed, num_samples=2560000)
    rand_dataloader = DataLoader(rand_dataset, batch_size=256, num_workers=10)
    embed_model = Embed_model(input_channels = 5, length = args.padded_length, 
                              output_dim = args.dimension, conv_channels = 256).to(device)
    tmodel = Twin_model(embed_model=embed_model,output_dim=args.dimension).to(device)

    tmodel.train()
    optimizer = torch.optim.Adam(tmodel.parameters(), lr=args.lr)

    loss_all = 0.
    running_loss = 0.
    loss_function_distance = PNLL
    loss_fucntion_distance_mse = MSE
    cnt = 0
    for _ in range(args.epochs):
        if _ == 2:
            optimizer.param_groups[0]['lr'] = args.lr*0.1
        print('LR: {}'.format(optimizer.param_groups[0]['lr']))
        for data,y in tqdm(rand_dataloader):
            cnt += 1
            y = y.to(device).to(torch.float32)
            data = data.to(device).to(torch.float32)
            optimizer.zero_grad()
            d = tmodel(data)

            loss_distance = loss_function_distance(d,y)
            loss_distance_mse = loss_fucntion_distance_mse(d,y)

            loss_value = loss_distance
            loss_value.backward()
            running_loss += loss_value.item()
            loss_all += loss_value.item()
            optimizer.step()

            if cnt % 400 == 0:
                print('#{}, loss {:.3f}, mse(<2) {:.3f}'.format(cnt, loss_distance.item(), loss_distance_mse.item()))

    embed_model_save_path = os.path.join(args.save_path,'embed_model.pth')
    torch.save(embed_model.state_dict(), embed_model_save_path)

if __name__ == '__main__':
    args = get_args()
    global device
    device = args.device

    train(args)
