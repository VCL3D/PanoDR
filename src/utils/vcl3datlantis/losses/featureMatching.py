import torch.nn as nn

class FeatureMatchingLoss(nn.Module):
    def __init__(self, criterion='l1'):
        super(FeatureMatchingLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)

    def forward(self, fake_features, real_features):
        """
           fake_features (list of lists): Discriminator features of fake images.
           real_features (list of lists): Discriminator features of real images.
        Returns:
        (tensor): Loss value.
        """
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j],
                                          real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss