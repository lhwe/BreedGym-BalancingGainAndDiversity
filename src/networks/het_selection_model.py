import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from src.networks.selection_model import CNNFeaturesExtractor


class HetCNNFeaturesExtractor(CNNFeaturesExtractor):
    """
    Same CNN backbone as CNNFeaturesExtractor, but also passes through the
    "heterozygosity" untouched (it doesn't need the CNN).
    """

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = CNNFeaturesExtractor._forward(self.shared_network, observations['obs'])
        return {
            'obs': features,
            'gen_number': observations['gen_number'],
            'heterozygosity': observations['heterozygosity'],
        }


class HetSelectionIndex(nn.Module):

    def __init__(
        self,
        features_dim,
        gen_features_dim=1,
        het_features_dim=8,
        policy_hiddens=[16],
        value_hiddens=[],
    ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_hiddens = [policy_hiddens] if isinstance(policy_hiddens, int) else (policy_hiddens or [])
        self.value_hiddens = [value_hiddens] if isinstance(value_hiddens, int) else (value_hiddens or [])

        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

        self.gen_projector_actor = torch.jit.script(
            nn.Sequential(nn.Linear(1, gen_features_dim), nn.LeakyReLU())
        )
        self.gen_projector_critic = torch.jit.script(
            nn.Sequential(nn.Linear(1, gen_features_dim), nn.LeakyReLU())
        )

        # Per-individual heterozygosity branch
        self.het_projector_actor = torch.jit.script(
            nn.Sequential(nn.Linear(1, het_features_dim), nn.LeakyReLU())
        )
        self.het_projector_critic = torch.jit.script(
            nn.Sequential(nn.Linear(1, het_features_dim), nn.LeakyReLU())
        )

        combined_dim = features_dim + gen_features_dim + het_features_dim

        policy_net = self._make_net(self.policy_hiddens, features_dim=combined_dim)
        policy_net.append(nn.Tanh())
        self.policy_net = torch.jit.script(policy_net)

        value_net = self._make_net(self.value_hiddens, features_dim=combined_dim)
        self.value_net = torch.jit.script(value_net)

    def _make_net(self, hiddens, features_dim) -> nn.Module:
        net = nn.Sequential()
        current_features = features_dim
        for n_features in hiddens:
            net.append(nn.Linear(current_features, n_features))
            current_features = n_features
            net.append(nn.ReLU())
        net.append(nn.Linear(current_features, 1))
        return net.to(self.device)

    def _expand_gen(self, gen_features, features):
        if len(features.shape) == 3:
            return gen_features.unsqueeze(1).expand(-1, features.shape[1], -1)
        num_inds = features.shape[0] // gen_features.shape[0]
        return gen_features.unsqueeze(1).expand(-1, num_inds, -1).reshape(features.shape[0], -1)

    def forward_actor(self, x):
        features = x['obs']
        gen_number = x['gen_number']
        het = x['heterozygosity']

        gen_features = self._expand_gen(self.gen_projector_actor(gen_number), features)
        # heterozygosity is already per-individual, shape matches `features` on
        # the batch/individual dims -- no expand needed, just project.
        het_features = self.het_projector_actor(het)

        out = self.policy_net(torch.cat([features, gen_features, het_features], dim=-1))
        batch_size = gen_number.shape[0]
        return out.view(batch_size, -1)

    def forward_critic(self, x):
        features = x['obs']
        gen_number = x['gen_number']
        het = x['heterozygosity']

        gen_features = self._expand_gen(self.gen_projector_critic(gen_number), features)
        het_features = self.het_projector_critic(het)

        out = self.value_net(torch.cat([features, gen_features, het_features], dim=-1))
        batch_size = gen_number.shape[0]
        return out.view(batch_size, -1).mean(dim=-1)

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class BiomarkerSelectionAC(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        gen_features_dim=1,
        het_features_dim=8,
        policy_hiddens=[16],
        value_hiddens=[],
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):
        self.policy_hiddens = policy_hiddens
        self.value_hiddens = value_hiddens
        self.gen_features_dim = gen_features_dim
        self.het_features_dim = het_features_dim

        super().__init__(
            observation_space, action_space, lr_schedule,
            net_arch, activation_fn, *args, **kwargs,
        )

        self.ortho_init = False
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HetSelectionIndex(
            features_dim=self.features_extractor.features_dim,
            gen_features_dim=self.gen_features_dim,
            het_features_dim=self.het_features_dim,
            policy_hiddens=self.policy_hiddens,
            value_hiddens=self.value_hiddens,
        )