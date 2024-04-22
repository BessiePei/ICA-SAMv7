import torch


pt_path = r'C:\Users\xxx\Desktop\yolov7-pytorch-master\nets\best.pt'


pth_path = r'C:\Users\xxx\Desktop\yolov7-pytorch-master\nets\best.pth'


model_weights = torch.load(pt_path, map_location='cpu')


torch.save(model_weights, pth_path)

print(f'The model weights have been successfully converted from {pt_path} to {pth_path}.')