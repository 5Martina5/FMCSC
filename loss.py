import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion1 = nn.CrossEntropyLoss()

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):
        batch = h_i.shape[0]
        N = 2 * batch
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, batch)
        sim_j_i = torch.diag(sim, -batch)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_feature1(self, h_i, h_j):
        batch = h_i.shape[0]
        N = 2 * batch
        h = torch.cat((h_i, h_j), dim=0) # 按行拼接；行数翻倍
        nmse_matrix = torch.zeros((N, N))
        for i in range(N):
            for j in range(N):
                nmse_matrix[i, j] = torch.nn.functional.mse_loss(h[i], h[j])
        sim = nmse_matrix / self.temperature_f
        sim_i_j = torch.diag(sim, batch)
        sim_j_i = torch.diag(sim, -batch)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_model(self, glob_h, h, zs):
        N = glob_h.size(0)
        pos = self.similarity(glob_h, h)
        logits = pos.reshape(-1, 1)
        nega = self.similarity(h, zs)
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.temperature_l
        labels = torch.zeros(glob_h.size(0)).cuda().long()

        loss = self.criterion1(logits, labels)
        # loss /= N
        return loss


