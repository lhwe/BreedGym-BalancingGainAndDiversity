import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from src.networks.het_selection_model import HetCNNFeaturesExtractor


class HetInteractionSelectionIndex(nn.Module):
    """Selection network with explicit heterozygosity-genotype interactions.

    Individual heterozygosity is standardized within each candidate population.
    Separate actor and critic branches project this relative heterozygosity and
    use it to gate the genotype representation. The original genotype features
    are retained, so the interaction is an additional residual signal rather
    than a replacement for the baseline representation.
    """

    def __init__(
        self,
        features_dim,
        gen_features_dim=16,
        het_features_dim=16,
        policy_hiddens=(32,),
        value_hiddens=(32,),
    ) -> None:
        super().__init__()

        self.policy_hiddens = self._as_list(policy_hiddens)
        self.value_hiddens = self._as_list(value_hiddens)

        # Stable-Baselines3 expects these attributes from the MLP extractor.
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

        self.gen_projector_actor = nn.Sequential(
            nn.Linear(1, gen_features_dim), nn.LeakyReLU()
        )
        self.gen_projector_critic = nn.Sequential(
            nn.Linear(1, gen_features_dim), nn.LeakyReLU()
        )
        self.het_projector_actor = nn.Sequential(
            nn.Linear(1, het_features_dim), nn.LeakyReLU()
        )
        self.het_projector_critic = nn.Sequential(
            nn.Linear(1, het_features_dim), nn.LeakyReLU()
        )

        # A heterozygosity-conditioned gate creates an explicit interaction
        # with every genotype feature. Tanh permits positive and negative
        # modulation and is zero-centred at initialization in expectation.
        self.het_gate_actor = nn.Sequential(
            nn.Linear(het_features_dim, features_dim), nn.Tanh()
        )
        self.het_gate_critic = nn.Sequential(
            nn.Linear(het_features_dim, features_dim), nn.Tanh()
        )

        # genotype + generation + heterozygosity + gated genotype interaction
        combined_dim = 2 * features_dim + gen_features_dim + het_features_dim
        self.policy_net = self._make_net(self.policy_hiddens, combined_dim)
        self.policy_net.append(nn.Tanh())
        self.value_net = self._make_net(self.value_hiddens, combined_dim)

    @staticmethod
    def _as_list(hiddens):
        if hiddens is None:
            return []
        if isinstance(hiddens, int):
            return [hiddens]
        return list(hiddens)

    @staticmethod
    def _make_net(hiddens, features_dim):
        layers = []
        current_dim = features_dim
        for hidden_dim in hiddens:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _expand_gen(gen_features, features):
        if features.ndim == 3:
            return gen_features.unsqueeze(1).expand(-1, features.shape[1], -1)
        num_individuals = features.shape[0] // gen_features.shape[0]
        return (
            gen_features.unsqueeze(1)
            .expand(-1, num_individuals, -1)
            .reshape(features.shape[0], -1)
        )

    @staticmethod
    def _standardize_heterozygosity(heterozygosity, batch_size):
        """Standardize across individuals, separately for every population."""
        original_shape = heterozygosity.shape
        if heterozygosity.ndim == 3:
            population_view = heterozygosity
        else:
            population_view = heterozygosity.reshape(batch_size, -1, 1)

        mean = population_view.mean(dim=1, keepdim=True)
        std = population_view.std(dim=1, keepdim=True, unbiased=False)
        standardized = (population_view - mean) / (std + 1e-6)
        return standardized.reshape(original_shape)

    def _combine(self, x, actor):
        features = x["obs"]
        gen_number = x["gen_number"]
        heterozygosity = self._standardize_heterozygosity(
            x["heterozygosity"], gen_number.shape[0]
        )

        if actor:
            gen_features = self.gen_projector_actor(gen_number)
            het_features = self.het_projector_actor(heterozygosity)
            interaction = features * self.het_gate_actor(het_features)
        else:
            gen_features = self.gen_projector_critic(gen_number)
            het_features = self.het_projector_critic(heterozygosity)
            interaction = features * self.het_gate_critic(het_features)

        gen_features = self._expand_gen(gen_features, features)
        return torch.cat(
            [features, gen_features, het_features, interaction], dim=-1
        )

    def forward_actor(self, x):
        out = self.policy_net(self._combine(x, actor=True))
        return out.view(x["gen_number"].shape[0], -1)

    def forward_critic(self, x):
        out = self.value_net(self._combine(x, actor=False))
        return out.view(x["gen_number"].shape[0], -1).mean(dim=-1)

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class HetInteractionSelectionAC(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        gen_features_dim=16,
        het_features_dim=16,
        policy_hiddens=(32,),
        value_hiddens=(32,),
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):
        self.policy_hiddens = policy_hiddens
        self.value_hiddens = value_hiddens
        self.gen_features_dim = gen_features_dim
        self.het_features_dim = het_features_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

        self.ortho_init = False
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HetInteractionSelectionIndex(
            features_dim=self.features_extractor.features_dim,
            gen_features_dim=self.gen_features_dim,
            het_features_dim=self.het_features_dim,
            policy_hiddens=self.policy_hiddens,
            value_hiddens=self.value_hiddens,
        )


__all__ = [
    "HetCNNFeaturesExtractor",
    "HetInteractionSelectionAC",
    "HetInteractionSelectionIndex",
]
