import numpy as np

def mutual_information(matrix1, matrix2, bins=10):
        flat_matrix1 = matrix1.flatten().cpu().detach().numpy()
        flat_matrix2 = matrix2.flatten().cpu().detach().numpy()
        hist_2d, _, _ = np.histogram2d(flat_matrix1, flat_matrix2, bins=bins)

        joint_prob = hist_2d / np.sum(hist_2d)

        marginal_prob1 = np.sum(hist_2d, axis=1) / np.sum(hist_2d)
        marginal_prob2 = np.sum(hist_2d, axis=0) / np.sum(hist_2d)

        joint_prob[joint_prob == 0] = 1e-10
        marginal_prob1[marginal_prob1 == 0] = 1e-10
        marginal_prob2[marginal_prob2 == 0] = 1e-10

        mi = np.sum(joint_prob * np.log2(joint_prob / (np.outer(marginal_prob1, marginal_prob2))))

        # mi_tensor = torch.tensor(mi, dtype=torch.float32)

        return mi



