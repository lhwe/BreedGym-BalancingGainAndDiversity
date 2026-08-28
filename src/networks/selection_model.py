import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


CONV1_DEFAULT = {"out_channels": 64, "kernel_size": 256, "stride": 32}
CONV2_DEFAULT = {"out_channels": 16, "kernel_size": 8, "stride": 2}

class CNNFeaturesExtractor(BaseFeaturesExtractor):    
    def __init__(
        self,
        observation_space,
        conv1_kwargs=CONV1_DEFAULT,
        conv2_kwargs=CONV2_DEFAULT,
        load_path=None,
        features_dim=None,
        use_gen_obs=None,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gen_obs = use_gen_obs

        obs_shape = observation_space['obs'].shape # box: observation_space.shape
        shared_network = nn.Sequential(
            nn.Conv1d(obs_shape[-1], **conv1_kwargs),
            nn.ReLU(),
            nn.Conv1d(conv1_kwargs["out_channels"], **conv2_kwargs),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)
        
        sample = torch.zeros(obs_shape, device=self.device)
        with torch.no_grad():
            out_sample = CNNFeaturesExtractor._forward(shared_network, sample)
        
        if features_dim is None:
            features_dim = out_sample.shape[-1]
        else:
            shared_network.append(nn.Linear(out_sample.shape[-1], features_dim))
            shared_network.append(nn.ReLU())
        
        if load_path is not None:
            state_dict = torch.load(load_path)
            model_dict = shared_network.state_dict()

            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            print(f"Loading {len(pretrained_dict)} layers from pretrained CNN...")
            model_dict.update(pretrained_dict)
            shared_network.load_state_dict(model_dict)
        
        super().__init__(observation_space, features_dim=features_dim)
        self.shared_network = torch.jit.script(shared_network)    

    @staticmethod
    def _forward(net, x):
        batch_pop = x.reshape(-1, x.shape[-2], x.shape[-1])
        batch_pop = batch_pop.permute(0, 2, 1)
        out1 = net(batch_pop)
        chan_indices = torch.arange(x.shape[-1])
        chan_indices[0] = 1
        chan_indices[1] = 0
        out2 = net(batch_pop[:, chan_indices])
        features = out1 + out2
        return features.reshape(*x.shape[:-2], -1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations['obs']
        features = CNNFeaturesExtractor._forward(self.shared_network, x)
        if self.use_gen_obs:
            return {'obs': features, 'gen_number': observations['gen_number']}
        else:
            return features


class SelectionIndex(nn.Module):

    def __init__(
        self,
        features_dim,
        gen_features_dim=1,
        policy_hiddens=[16],
        value_hiddens=[],
        use_gen_obs=None,
    ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if policy_hiddens is None:
            self.policy_hiddens = []
        elif isinstance(policy_hiddens, int):
            self.policy_hiddens = [policy_hiddens]
        else:
            self.policy_hiddens = policy_hiddens

        if value_hiddens is None:
            self.value_hiddens = []
        elif isinstance(value_hiddens, int):
            self.value_hiddens = [value_hiddens]
        else:
            self.value_hiddens = value_hiddens

        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

        self.use_gen_obs = use_gen_obs
        if self.use_gen_obs:
            self.gen_projector_actor = torch.jit.script(
                nn.Sequential(
                    nn.Linear(1, gen_features_dim),
                    nn.LeakyReLU()
                )
            )
            self.gen_projector_critic = torch.jit.script(
                nn.Sequential(
                    nn.Linear(1, gen_features_dim),
                    nn.LeakyReLU()
                )
            )
            mlp_input_dim = features_dim + gen_features_dim
        else:
            mlp_input_dim = features_dim

        policy_net = self._make_net(self.policy_hiddens, features_dim=mlp_input_dim)
        policy_net.append(nn.Tanh())
        self.policy_net = torch.jit.script(policy_net)

        value_net = self._make_net(self.value_hiddens,  features_dim=mlp_input_dim)
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

    def _process_input(self, x):
        if self.use_gen_obs:
            features = x['obs']
            gen_number = x['gen_number']
            return features, gen_number
        else:
            return x, None

    def forward_actor(self, x):
        features, gen_number = self._process_input(x)
        if self.use_gen_obs:
            gen_features = self.gen_projector_actor(gen_number)
            if len(features.shape) == 3: # if it is 3d: [Batch, Individuals, 64]
                gen_features = gen_features.unsqueeze(1).expand(-1, features.shape[1], -1)
            else: # if it is flatten: [Batch * Individuals, 64]
                num_inds = features.shape[0] // gen_number.shape[0]
                gen_features = gen_features.unsqueeze(1).expand(-1, num_inds, -1).reshape(features.shape[0], -1)
            combined = torch.cat([features, gen_features], dim=-1)
        else:
            combined = features

        out = self.policy_net(combined)
        batch_size = features.shape[0] if self.use_gen_obs else x.shape[0]
        return out.view(batch_size, -1)
        # return out.squeeze()

    def forward_critic(self, x):
        features, gen_number = self._process_input(x)

        if self.use_gen_obs:
            gen_features = self.gen_projector_critic(gen_number)
            if len(features.shape) == 3:
                gen_features = gen_features.unsqueeze(1).expand(-1, features.shape[1], -1)
            else:
                num_inds = features.shape[0] // gen_number.shape[0]
                gen_features = gen_features.unsqueeze(1).expand(-1, num_inds, -1).reshape(features.shape[0], -1)
            combined = torch.cat([features, gen_features], dim=-1)
        else:
            combined = features

        out = self.value_net(combined)
        batch_size = features.shape[0] if self.use_gen_obs else x.shape[0]
        return out.view(batch_size, -1).mean(dim=-1)

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class SelectionAC(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        gen_features_dim=1,
        policy_hiddens=[16],
        value_hiddens=[],
        activation_fn=nn.Tanh,
        use_gen_obs=None,
        *args,
        **kwargs,
    ):
        # infer generation-aware architecture from the actual observation space
        if use_gen_obs is None:
            if hasattr(observation_space, "spaces"):
                use_gen_obs = "gen_number" in observation_space.spaces
            else:
                use_gen_obs = False

        self.policy_hiddens = policy_hiddens
        self.value_hiddens = value_hiddens
        self.gen_features_dim = gen_features_dim
        self.use_gen_obs = use_gen_obs

        features_extractor_kwargs = dict(kwargs.get("features_extractor_kwargs") or {})
        features_extractor_kwargs["use_gen_obs"] = use_gen_obs
        kwargs["features_extractor_kwargs"] = features_extractor_kwargs

        super(SelectionAC, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

        # Disable orthogonal initialization
        self.ortho_init = False

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SelectionIndex(
            features_dim=self.features_extractor.features_dim,
            gen_features_dim=self.gen_features_dim,
            policy_hiddens=self.policy_hiddens,
            value_hiddens=self.value_hiddens,
            use_gen_obs=self.use_gen_obs,
        )