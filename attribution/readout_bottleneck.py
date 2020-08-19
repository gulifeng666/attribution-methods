import torch.nn as nn
from attribution_bottleneck.attribution.base import AttributionMethod
from attribution_bottleneck.bottleneck.readout_bottleneck import ReadoutBottleneck
from attribution_bottleneck.utils.misc import resize, replace_layer, to_np
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
class ReadoutBottleneckReader(AttributionMethod):

    def __init__(self, model, target_layer, bn_layer: ReadoutBottleneck):
        super().__init__()

        self.bn_layer = bn_layer
        self.target_layer = target_layer
        self.sequential = nn.Sequential(target_layer, bn_layer)
        self.model = model

    def _inject_bottleneck(self):
        replace_layer(self.model, self.target_layer, self.sequential)
        self.bn_layer.active = True

    def _remove_bottleneck(self):
        replace_layer(self.model, self.sequential, self.target_layer)
        self.bn_layer.active = False
    def few_shot_heatmap(self,model,x,iter,tag,n_support = 1):

        heatmaps = self.heatmap(x.reshape(-1, x.shape[2], x.shape[3], x.shape[4]), [])

        #          heatmaps[i,j] = torch.from_numpy(reader.heatmap(x[i,j].squeeze(0)))
        # summary.add_image(tag+'heatmapresult', heatmaps, iter, dataformats='HW')
        normal_x = (x - x.min()) / (x.max() - x.min())
        grid = make_grid(normal_x.view(-1, 3, 224, 224), nrow=x.shape[1])
        self.bn_layer.summary.add_image(tag + 'imresult', grid, iter, dataformats='CHW')
        figure = plt.figure(figsize=(4, 10))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                plt.subplot(x.shape[0], x.shape[1], i * x.shape[1] + j + 1)
                plt.imshow(normal_x[i, j].cpu().permute(1, 2, 0))
                if (j == n_support):
                    plt.title('query', fontsize=10)
                else:
                    plt.title('support', fontsize=10)
                # heatmaps[i,j] = torch.from_numpy(reader.heatmap(x[i,j].squeeze(0)))
                plt.axis('off')
        plt.tight_layout()
        self.bn_layer.summary.add_figure(tag + 'imageresult', figure, iter)
        scores = model.set_forward(x)
        figure = plt.figure(figsize=(4, 10))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                plt.subplot(x.shape[0], x.shape[1], i * x.shape[1] + j + 1)
                plt.imshow(heatmaps[i * x.shape[1] + j], cmap="jet")
                # plt.imshow(heatmaps[ i * x.shape[1] + j ], cmap="jet")
                if (j == n_support):
                    plt.title('score:' + str(np.around((nn.Softmax(0)(scores[i])).detach().cpu().numpy(), decimals=2)),
                              fontsize=5)
                else:
                    pass
                plt.axis('off')
        plt.tight_layout()
        self.bn_layer.summary.add_figure(tag + 'heatmapresult', figure, iter)
        grid = make_grid(torch.tensor(heatmaps).view(-1, 1, 224, 224), nrow=x.shape[1])
        self.bn_layer.summary.add_image(tag + 'hpresult', grid, iter, dataformats='CHW')

    # scores = self.set_forward(x)
    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):

        self.model.eval()
        self._inject_bottleneck()
        with torch.no_grad():
            self.model(input_t)
        self._remove_bottleneck()
        if(input_t.shape[0]>1):
            hmaps = np.zeros((input_t.shape[0],input_t.shape[2],input_t.shape[3]))
            for i in range(input_t.shape[0]):
              htensor = to_np(self.bn_layer.buffer_capacity[i])

              hmap = htensor.mean(0) if(len(htensor.shape)==3) else htensor
              hmap = resize(hmap, input_t.shape[2:])

              hmap = hmap - hmap.min()
              hmap = hmap/(max(hmap.max(),1e-5))
              hmaps[i] = hmap
            hmap = hmaps
        else:
            htensor = to_np(self.bn_layer.buffer_capacity)
            hmap = htensor.mean(axis=(0, 1))
            hmap = resize(hmap, input_t.shape[2:])
            hmap = hmap - hmap.min()
            hmap = hmap / (max(hmap.max(), 1e-5))

        return hmap
