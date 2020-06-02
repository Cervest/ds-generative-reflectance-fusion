import torch
import torch.nn as nn


class GANLoss(nn.Module):

    def __init__(self, lbda):
        super().__init__()
        self.lbda = lbda
        self.cross_entropy = nn.BCELoss()

    def forward(self, output_real_sample, output_fake_sample):
        """Adversarial networks loss computation given by :
            LossDisc = E_{x~realdata}[-logD(x)] + E_{z~inputs}[-log(1 - D(G(z)))]
            LossGen = E_{z~inputs}[-logD(z)]
            We approximate:
                E_{x~realdata}[-logD(x)] = Avg(CrossEnt_{x:realbatch}(1, D(x)))
                E_{z~inputs}[-log(1 - D(G(z)))] = Avg(CrossEnt_{x:fakebatch}(0, D(x)))
                E_{z~inputs}[-logD(z)] = Avg(CrossEnt_{x:fakebatch}(1, D(x)))
        Args:
            output_real_sample (torch.Tensor): (N, ) discriminator prediction on real samples
            output_fake_sample (torch.Tensor): (N, ) discriminator prediction on fake samples
        """
        # Setup targets vectors
        target_real_sample = torch.ones_like(output_real_sample)
        target_fake_sample = torch.zeros_like(output_fake_sample)

        # Losses computation, criterion should be crossentropy
        loss_real_sample = self.cross_entropy(output_real_sample, target_real_sample)
        loss_fake_sample = self.cross_entropy(output_fake_sample, target_fake_sample)
        disc_loss = loss_real_sample + loss_fake_sample
        gen_loss = self.cross_entropy(output_fake_sample, target_real_sample)
        return gen_loss, disc_loss

    @property
    def lbda(self):
        return self._lbda

    @lbda.setter
    def lbda(self, lbda):
        self._lbda = lbda
