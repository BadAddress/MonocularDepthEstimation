from ____model import*
from data_revized import* 
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_ssim import SSIM



def compute_metrics(pred_batch,gt_batch):
    #   432,576
    abs_diff, abs_rel, log10, a1, a2, a3,rmse_tot,rmse_log_tot = 0,0,0,0,0,0,0,0
    batch_size = gt_batch.size(0)
    for gt, pred in zip(gt_batch, pred_batch):

        thresh = torch.max((gt / pred), (pred / gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()
        rmse = (gt - pred) ** 2
        rmse_tot += torch.sqrt(torch.mean(rmse))
        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        rmse_log_tot += torch.sqrt(torch.mean(rmse_log))
        abs_diff += torch.mean(torch.abs(gt - pred))
        abs_rel += torch.mean(torch.abs(gt - pred) / gt)
        log10 += torch.mean(torch.abs(torch.log10(gt)-torch.log10(pred)))

    metrics_dict = {
        'abs_diff': abs_diff.item() / batch_size,
        'abs_rel': abs_rel.item() / batch_size,
        'log10': log10.item() / batch_size,
        'a1': a1.item() / batch_size,
        'a2': a2.item() / batch_size,
        'a3': a3.item() / batch_size,
        'rmse_tot': rmse_tot.item() / batch_size,
        'rmse_log_tot': rmse_log_tot.item() / batch_size
    }


    return metrics_dict







def train_run():

    state_of_art = binNet()

    nyu = MyDataset("train")
    dataloader = DataLoader(nyu,batch_size=6,shuffle=False)
    state_of_art = state_of_art.to('cuda')
    state_of_art.train()
    optimizer = torch.optim.SGD(state_of_art.parameters(), lr=0.005, momentum=0.9)

    epoch_num = 200
    minLoss = 1000
    log_file_path = '/home/bl/Desktop/bde/train_logs/myLogs.txt'
    
    huber_loss = nn.SmoothL1Loss()
    
    for epoch in range(epoch_num):
        if epoch > 100:  
            w1, w2, w3 = 0.2, 0.2, 0.6 
        print("Current Epoch: ",epoch)
        batchCnt = 0
        for data in tqdm(dataloader):
            vp_tensor = data[0].to('cuda')
            seg_tensor = data[1].to('cuda')
            rgb_tensor = data[2].to('cuda')
            gt_tensor = data[3].to('cuda')
            pred_tensor = state_of_art(vp_tensor,seg_tensor,rgb_tensor)

            valid_gt = gt_tensor.clamp(1e-3, 10)
            valid_pred = pred_tensor.clamp(1e-3,10)

            loss = state_of_art.scale_invariant_loss(valid_pred,valid_gt)
            metrics = compute_metrics(valid_pred,valid_gt)

            l1_loss = torch.mean(torch.abs(valid_pred - valid_gt))  
            mse_loss = torch.nn.functional.mse_loss(valid_pred, valid_gt)
            hloss = huber_loss(valid_pred,valid_gt)
            # print(loss.item(),l1_loss.item(),mse_loss.item(),hloss.item())
            totalLoss = 0.01*loss + 0.2*l1_loss + 0.5*hloss 

            # batchLoss = batchLoss
            # metrics['batchLoss'] = batchLoss

            log_message = f"[epoch_{epoch}][batch_{batchCnt}] " + " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

            if batchCnt % 1000 ==0:
                if loss < minLoss:
                    torch.save(state_of_art.state_dict(), 'best_model.pth')
            
            if batchCnt % 500 == 0:
                debugTool.tensorViz([vp_tensor.detach(),
                        seg_tensor.detach(),
                        gt_tensor.detach(),
                        rgb_tensor.detach(),
                        pred_tensor.detach()],epoch,batchCnt)
            
            if batchCnt % 100 ==0:
                # 将日志消息追加到文件
                with open(log_file_path, 'a') as log_file:
                    log_file.write(log_message + "\n")
        
            
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()
            batchCnt += 1

            







if __name__ == "__main__":
    train_run()
