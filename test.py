import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#返回cuda表示成功
#或者
print(torch.cuda.is_available())
#返回True表示成功
