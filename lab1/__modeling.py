import torch


class SiameseNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encode = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32)
        )

        self.act = torch.nn.Sigmoid()
    
    @torch.compile(fullgraph=True, mode='max-autotune')
    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        enc_1 = self.encode(input_1)
        enc_2 = self.encode(input_2)

        enc_1 = enc_1 / (enc_1.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        enc_2 = enc_2 / (enc_2.norm(p=2, dim=-1, keepdim=True) + 1e-13)

        sims = torch.einsum(
            'ij,kj->ik',
            enc_1,
            enc_2
        )

        return self.act(sims)