import torch
import audtorch
import torch.nn.functional as F

class corr_loss(torch.nn.Module):

    def __init__(self):
        super(corr_loss, self).__init__()

    def forward(self, x, y):
        b, c, h, w = y.size()

        A = torch.zeros([b, c, c]).float().cuda()
        for i in range(b):
            r = torch.squeeze(y[i, 0, :, :])
            g = torch.squeeze(y[i, 1, :, :])
            b = torch.squeeze(y[i, 2, :, :])
            histr = torch.histc(r, 25, 0.0, 1.0)
            histg = torch.histc(g, 25, 0.0, 1.0)
            histb = torch.histc(b, 25, 0.0, 1.0)

            crg = audtorch.metrics.functional.pearsonr(histr, histg)
            crb = audtorch.metrics.functional.pearsonr(histr, histb)
            cgb = audtorch.metrics.functional.pearsonr(histg, histb)

            A[i, :, :] = torch.tensor([[1, crg, crb], [crg, 1, cgb], [crb, cgb, 1]]).float().cuda()

        # Qx = torch.einsum('bchw,cmn->bmnhw', x, Q) # x has shape (batch, channel, height, width)
        # Qy = torch.einsum('bchw,cmn->bmnhw', y, Q)
        d = x - y
        Qd = torch.einsum('bcm,bmhw->bchw', A, d)


        #ry = 1.2-y
        #Qr = torch.einsum('bchw,cmn->bmnhw', ry, Q)
        #QD = torch.einsum('bmnhw,bnlhw->bmlhw', Qd, Qr)  # Qd*(1-y)

        #r = torch.abs(d/(y+1e-5))
        #Qr = torch.einsum('bchw,cmn->bmnhw', r, Q)
        #QD = torch.einsum('bmnhw,bnlhw->bmlhw', Qd, Qr)  # Qd*Qr

        #Qdt = torch.einsum('bmnhw->bnmhw', Qd)
        #QD = torch.einsum('bmnhw,bnlhw->bmlhw', Qdt, Qd) #transpose(A-B)*(A-B)

        #Qdt = Qx+Qy
        #QD = torch.einsum('bmnhw,bnlhw->bmlhw', Qdt, Qd) #(A + B) * (A - B)
        alpha = 0.8
        beta = 0.2
        corr = torch.norm(Qd, p=1)/torch.numel(Qd)
        l1 = F.l1_loss(x, y)
        loss = alpha*l1 + beta*corr

        return loss
