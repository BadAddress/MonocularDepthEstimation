
import matplotlib.pyplot as plt
import torch
import numpy as np


def tensorViz(tensorList,epoch_n,batch_n,option='save'):
    """ tensorList params 
          dist_theta_tensor, seg_tensor, gt_tensor, rgb_tensor, pred_tensor
    """

    dist_theta_tensor = tensorList[0][0].cpu()
    seg_tensor = tensorList[1][0].cpu()
    gt_tensor = tensorList[2][0].cpu()
    rgb_tensor = tensorList[3][0].cpu()
    pred_tensor = tensorList[4][0].cpu()

     
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # 调整布局和大小

    # 显示RGB张量
    axes[0, 0].imshow(rgb_tensor.permute(1, 2, 0))  # 调整通道顺序为H x W x C
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis('off')

    # 显示深度GT张量
    axes[0, 1].imshow(gt_tensor.squeeze(), cmap='jet')  # 假设深度GT是单通道的
    axes[0, 1].set_title("Depth Ground Truth")
    axes[0, 1].axis('off')

    # 显示预测深度张量
    axes[0, 2].imshow(pred_tensor.squeeze(), cmap='jet')  # 同样假设预测深度是单通道的
    axes[0, 2].set_title("Predicted Depth")
    axes[0, 2].axis('off')

    dist = dist_theta_tensor[0, :, :]
    theta = dist_theta_tensor[1, :, :]

    axes[1, 0].imshow(dist.squeeze(), cmap='jet')
    axes[1, 0].set_title("Vanishing Point: Distance")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(theta.squeeze(), cmap='jet')
    axes[1, 1].set_title("Vanishing Point: Theta")
    axes[1, 1].axis('off')

    # 显示语义分割张量
    axes[1, 2].imshow(seg_tensor.squeeze(), cmap='jet')
    axes[1, 2].set_title("Semantic Segment")
    axes[1, 2].axis('off')

    # 隐藏剩余的空白子图
    for ax in axes[2, :]:
        ax.axis('off')

    # 根据选项保存或显示图像
    if option == 'save':
        save_path = f'/home/bl/Desktop/bde/train_viz/epoch{epoch_n}_batch{batch_n}_res.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    elif option == 'show':
        plt.show()





























# def visualize_tensors(dist_theta_tensor, seg_tensor, gt_tensor, rgb_tensor):
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 调整大小以适应您的需要

#     # gt_tensor = gt_tensor/1000
#     # gt_tensor[gt_tensor>10] = 10.0

#     # 显示RGB张量
#     axes[0, 0].imshow(rgb_tensor.permute(1, 2, 0))  # 调整通道顺序为H x W x C
#     axes[0, 0].set_title("Original RGB")
#     axes[0, 0].axis('off')

#     # 显示深度GT张量
#     axes[0, 1].imshow(gt_tensor.squeeze(), cmap='jet')  # 假设深度GT是单通道的
#     axes[0, 1].set_title("Depth Ground Truth")
#     axes[0, 1].axis('off')

#     # 这里假设你的数据范围在一定范围内，你可能需要根据实际数据调整颜色映射和合并方式

#     dist= dist_theta_tensor[0,:, :]
#     theta= dist_theta_tensor[1,:, :]


#     axes[1,0].imshow(dist.squeeze(), cmap='jet')
#     axes[1, 0].set_title("Vanishing Point:distance")
#     axes[1, 0].axis('off')

#     axes[1,1].imshow(theta.squeeze(),cmap='jet')
#     axes[1, 0].set_title("Vanishing Point:theta")
#     axes[1,1].axis('off')

#     # 显示语义分割张量
#     axes[1, 2].imshow(seg_tensor.squeeze(), cmap='hot')
#     axes[1, 2].set_title("Semantic Segment")
#     axes[1, 2].axis('off')

#     # 隐藏空白的第三个子图
#     axes[0, 2].axis('off')

#     plt.show()

# 示例用，您应该用实际的张量替换下面的示例张量
# tensor1 = torch.randn(1, H, W)
# tensor2 = torch.randn(1, H, W)
# tensor3 = torch.randn(1, H, W)
# tensor_rgb = torch.randn(3, H, W)
# visualize_tensors(tensor1, tensor2, tensor3, tensor_rgb)
