# %%
import cv2
import sys
import torch
from torchsummary import summary

import numpy as np

from tqdm.auto import tqdm
from pathlib import Path
from images import file_names, Image, plt

# %%

if len(sys.argv) == 1:
    image = Image(file_names[2])
else:
    image = Image(Path(sys.argv[1]))

# %%
num = 137
sz = (28, 28)


def fetch_random(image, num=num, sz=sz):
    assert isinstance(image, Image), 'Input image is not Image'
    assert sz[0] < image.shape[0] and sz[1] < image.shape[1], 'Input sz is too large, {}, {}'.format(
        sz, image.shape)
    X = []
    y = []
    for _ in range(num):
        a = np.random.randint(0, image.shape[0]-sz[0])
        b = np.random.randint(0, image.shape[1]-sz[1])

        c, h, r, g = image.get((a, b, sz[0], sz[1]))
        X.append(g[np.newaxis, :])
        y.append(r[0][0])

    X = np.array(X, dtype=np.float32) / 255
    y = np.array(y, dtype=np.float32) / 255

    return X, y


def fetch_all(image, sz=sz):
    assert sz[0] < image.shape[0] and sz[1] < image.shape[1], 'Input sz is too large, {}, {}'.format(
        sz, image.shape)

    X = []
    p = []
    for a in range(image.shape[0] - sz[0]):
        for b in range(image.shape[1] - sz[1]):
            c, h, r, g = image.get((a, b, sz[0], sz[1]))
            X.append(g[np.newaxis, :])
            p.append((a, b))

    X = np.array(X, dtype=np.float32) / 255

    return X, p


X, y = fetch_random(image)
X.shape, y.shape

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%


class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64, 100)
        self.mlp2 = torch.nn.Linear(100, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


model = CNNnet().to(device)
print(model)

summary(model, input_size=(1, 28, 28))

# %%
param_path = Path('parameters/{}.param'.format(image.path.name))

if param_path.is_file():
    model.load_state_dict(torch.load(param_path))
    model.eval()
else:
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_count = []

    for repeat in tqdm(range(10000)):
        X, y = fetch_random(image)

        out = model(torch.tensor(X).to(device))
        loss = loss_func(out, torch.tensor(y).to(device))

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_count.append(loss.item())
        if repeat % 100 == 0:
            print('{}:\t'.format(repeat), loss.item())

    plt.figure('PyTorch_CNN_Loss')
    plt.plot(loss_count, label='Loss')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), param_path)


# %%
for name in file_names:
    print(name)

    image1 = Image(name)

    raw_file = Path('converted/{}.jpg'.format(image1.path.name))
    cv2.imwrite(raw_file.as_posix(), image1.color)

    out_file = Path(
        'converted/{}.{}.jpg'.format(image.path.name, image1.path.name))

    if out_file.is_file():
        continue

    X, pos = fetch_all(image1)

    rgb = []

    for j in tqdm(range(0, len(X), 10000), name.as_posix()):
        out = model(torch.tensor(X[j:j+10000]).to(device))
        arr = out.cpu().detach().numpy()
        rgb.append(arr)

    rgb = np.concatenate(rgb, axis=0)

    print(rgb.shape)

    sz = np.max(pos, axis=0)
    mat = image1.rgb.copy()

    for p, r in tqdm(zip(pos, rgb)):
        mat[p[0], p[1]] = r * 255

    cv2.imwrite(out_file.as_posix(), cv2.cvtColor(mat, cv2.COLOR_RGB2BGR))

    print('Saved new files into {}, {}'.format(out_file, raw_file))


# %%
# if __name__ == '__main__':
#     plt.imshow(mat)
#     plt.show()

#     image.plot()
#     plt.tight_layout()
#     plt.show()

#     image1.plot()
#     plt.tight_layout()
#     plt.show()

# %%
