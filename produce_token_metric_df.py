
# %%
# Setup
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
from neel.imports import *
from neel_plotly import *

# %%
import neel.utils as nutils

torch.set_grad_enabled(False)
SEED = 43
torch.manual_seed(SEED)
np.random.seed(SEED)

# %%
n_ctx = 256
DEVICE = "cpu"
# DATASET_NAME = "NeelNanda/pile-10k"
# MODEL_NAME = "pythia-70m"
DATASET_NAME = "stas/openwebtext-10k"
MODEL_NAME = "gpt2-small"
num_prompts = 1024
batch_size = 32
# %%


model = HookedTransformer.from_pretrained(MODEL_NAME)
dataset = load_dataset(DATASET_NAME, split="train")
token_dataset = utils.tokenize_and_concatenate(
    dataset, model.tokenizer, max_length=n_ctx
).shuffle(42)
pile_tokens = token_dataset["tokens"].cuda()
print(f"pile_tokens.shape: {pile_tokens.shape}")
print(f"pile_tokens first: {model.to_string(pile_tokens[0, :30])}")

W_OU = einops.einsum(
    model.W_O,
    model.W_U,
    "layer head d_head d_model, d_model d_vocab -> layer head d_head d_vocab",
)
print("W_OU.shape:", W_OU.shape)

n_layers = model.cfg.n_layers
# n_ctx = model.cfg.n_ctx
n_heads = model.cfg.n_heads
d_model = model.cfg.d_model
d_vocab = model.cfg.d_vocab
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp

# %% [markdown]
# ## Metrics

# %%
# Mover metrics
def get_head_dla(cache: ActivationCache, tokens: torch.Tensor):
    z = cache.stack_activation("z")
    z = z[:, :, :-1, :, :]
    # print("z.shape", z.shape) # [layer, batch, pos, head, d_head]
    W_OU_tokens = W_OU[:, :, :, tokens[:, 1:]]
    W_OU_tokens_scaled = W_OU_tokens / cache["scale"][:, :-1, 0]

    # print("W_OU_tokens_scaled.shape", W_OU_tokens_scaled.shape) # [layer head d_head batch pos]
    head_dla = einops.einsum(
        z,
        W_OU_tokens_scaled,
        "layer batch pos head d_head, layer head d_head batch pos -> layer head batch pos",
    )
    return head_dla


def max_prev_attended_to_token(cache):
    patterns = cache.stack_activation("pattern")
    # print("patterns.shape", patterns.shape) #[layer, batch head dest_pos src_pos]
    max_with_bos, argmax_with_bos = patterns.max(dim=-1)
    argmax_with_bos = einops.rearrange(
        argmax_with_bos, "layer batch head dest_pos -> layer head batch dest_pos"
    )[..., :-1]
    max_with_bos = einops.rearrange(
        max_with_bos, "layer batch head dest_pos -> layer head batch dest_pos"
    )[..., :-1]
    patterns[:, :, :, :, 0] = 0.0
    max_without_bos, argmax_without_bos = patterns.max(dim=-1)
    argmax_without_bos = einops.rearrange(
        argmax_without_bos, "layer batch head dest_pos -> layer head batch dest_pos"
    )[..., :-1]
    max_without_bos = einops.rearrange(
        max_without_bos, "layer batch head dest_pos -> layer head batch dest_pos"
    )[..., :-1]

    return argmax_with_bos, argmax_without_bos, max_with_bos, max_without_bos


def argmax_attn_to_token(argmax_token_index, tokens):
    predicted_token = torch.stack(
        [tokens[i][argmax_token_index[:, :, i, :]] for i in range(len(tokens))], dim=2
    )
    return predicted_token


def get_mover_attn_score(cache: ActivationCache, tokens: torch.Tensor):
    """
    Return the average attention paid to copies of the next token
    """
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]

    patterns = cache.stack_activation("pattern")
    is_next_token = tokens[:, 1:, None] == tokens[:, None, :-1]
    filtered_pattern = patterns[:, :, :, :-1, :-1] * is_next_token[None, :, None, :, :]
    return einops.reduce(
        filtered_pattern,
        "layer batch head dest_pos src_pos -> layer head batch dest_pos",
        "sum",
    )


def get_mover_dla_score(cache: ActivationCache, tokens: torch.Tensor):
    """
    Return the average attention paid to copies of the next token
    """
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]

    patterns = cache.stack_activation("pattern")
    is_next_token = tokens[:, 1:, None] == tokens[:, None, :-1]
    filtered_pattern = patterns[:, :, :, :-1, :-1] * is_next_token[None, :, None, :, :]
    v = cache.stack_activation("v")[:, :, :-1, :, :]
    filtered_z = einops.einsum(
        v,
        filtered_pattern,
        "layer batch src_pos head d_head, layer batch head dest_pos src_pos -> layer batch dest_pos head d_head",
    )

    # print("z.shape", z.shape) # [layer, batch, pos, head, d_head]
    W_OU_tokens = W_OU[:, :, :, tokens[:, 1:]]
    W_OU_tokens_scaled = W_OU_tokens / cache["scale"][:, :-1, 0]

    # print("W_OU_tokens_scaled.shape", W_OU_tokens_scaled.shape) # [layer head d_head batch pos]
    # print(filtered_z.shape)
    # print(W_OU_tokens_scaled.shape)
    mover_head_dla = einops.einsum(
        filtered_z,
        W_OU_tokens_scaled,
        "layer batch pos head d_head, layer head d_head batch pos -> layer head batch pos",
    )
    return mover_head_dla

# %%
# %%
# Actually running the model on the data distribution
torch.set_grad_enabled(False)
head_dla_list = []
mover_attn_score_list = []
mover_dla_score_list = []
plps_list = []

argmax_without_bos_list = []
max_without_bos_list = []
predicted_token_list = []

for i in tqdm.tqdm(range(0, num_prompts, batch_size)):
    tokens = pile_tokens[i : i + batch_size]
    logits, cache = model.run_with_cache(tokens)
    plps_list.append(model.loss_fn(logits, tokens, per_token=True).to(DEVICE))
    head_dla = get_head_dla(cache, tokens)
    head_dla_list.append(head_dla.to(DEVICE))
    mover_attn_score_list.append(get_mover_attn_score(cache, tokens).to(DEVICE))
    mover_dla_score_list.append(get_mover_dla_score(cache, tokens).to(DEVICE))

    _, argmax_without_bos, _, max_without_bos = max_prev_attended_to_token(cache)
    argmax_without_bos_list.append(argmax_without_bos.to(DEVICE))
    max_without_bos_list.append(max_without_bos.to(DEVICE))
    predicted_token = argmax_attn_to_token(argmax_without_bos, tokens)
    predicted_token_list.append(predicted_token.to(DEVICE))


# %%
plps = torch.cat(plps_list, dim=0)
head_dla = torch.cat(head_dla_list, dim=2)
print(head_dla.shape)
mover_attn_score = torch.cat(mover_attn_score_list, dim=2)
mover_dla_score = torch.cat(mover_dla_score_list, dim=2)
ratio_dla = mover_dla_score / head_dla
mover_dla = mover_dla_score
mover_attn = mover_attn_score

head_dla_flat = einops.rearrange(
    head_dla, "layer head batch pos -> (layer head) (batch pos)"
)
mover_attn_flat = einops.rearrange(
    mover_attn_score, "layer head batch pos -> (layer head) (batch pos)"
)
mover_dla_flat = einops.rearrange(
    mover_dla_score, "layer head batch pos -> (layer head) (batch pos)"
)
ratio_dla_flat = mover_dla_flat/head_dla_flat
# %%

argmax_without_bos = torch.cat(argmax_without_bos_list, dim=2)
print("{argmax_without_bos.shape=}")
max_without_bos = torch.cat(max_without_bos_list, dim=2)
predicted_token = torch.cat(predicted_token_list, dim=2)

# %%
head_df = pd.DataFrame(
    {
        "L": [l for l in range(n_layers) for h in range(n_heads)],
        "H": [h for l in range(n_layers) for h in range(n_heads)],
        "label": model.all_head_labels(),
    }
)

nutils.add_to_df(head_df, "head_dla", head_dla_flat.quantile(0.9, dim=-1))
nutils.add_to_df(head_df, "mover_attn", mover_attn_flat.quantile(0.9, dim=-1))
nutils.add_to_df(head_df, "mover_dla", mover_dla_flat.quantile(0.9, dim=-1))
nutils.add_to_df(head_df, "dla_ratio", (mover_dla_flat / head_dla_flat).quantile(0.9, dim=-1))
head_df.style.background_gradient("coolwarm")

# %%
ind_head_scores = nutils.get_induction_scores(model, make_plot=True)
_ = nutils.add_to_df(head_df, "induction", ind_head_scores)
# %%
embed = model.W_E
# line(embed[15])
post_mlp_embed = model.blocks[0].mlp(model.blocks[0].ln2(embed[None])).squeeze(0) + embed

eigenvals_resid = model.OV.eigenvalues
eigenvals_resid_score = ((eigenvals_resid.sum(-1)/eigenvals_resid.abs().sum(-1)).real)
imshow(eigenvals_resid_score)
eigenvals_vocab = (post_mlp_embed @ model.OV @ model.W_U).eigenvalues
eigenvals_vocab_score = ((eigenvals_vocab.sum(-1)/eigenvals_vocab.abs().sum(-1)).real)
imshow(eigenvals_vocab_score)

nutils.add_to_df(head_df, "eigenvals_vocab", eigenvals_vocab_score)
_ = nutils.add_to_df(head_df, "eigenvals_resid", eigenvals_resid_score)
nutils.show_df(head_df)
# %% [markdown]
# ## Dataset Filtering


token_subset = pile_tokens[:num_prompts]
num_copies = torch.tril(token_subset[:, :, None] == token_subset[:, None, :], -1).sum(-1)[:, 1:]
is_movable = num_copies > 0
imshow(is_movable[:20], title="Is Movable", xaxis="Pos", yaxis="Prompt")
imshow(num_copies[:20], title="Num Copies", xaxis="Pos", yaxis="Prompt")
print("Frac Movable:", is_movable.float().mean().item())

# %% [markdown]
# ### Induction Mask

# %%
# head_df.to_csv("head_df.csv")
def make_induction_mask(tokens, device=DEVICE):
    tokens = tokens.to(device)
    equality_check = tokens[:-1, None] == tokens[None, :-1]
    next_equality_check = tokens[1:, None] == tokens[None, 1:]
    return torch.tril(equality_check * next_equality_check, diagonal=-1).any(dim=-1)
make_induction_mask = torch.vmap(make_induction_mask)
induction_mask = make_induction_mask(token_subset)
print(induction_mask.shape, induction_mask.size())

# %% [markdown]
# ## No Induction Metrics

# %%
q = 0.95
nutils.add_to_df(head_df, "head_dla_no_ind", head_dla_flat[:, ~induction_mask.flatten()].quantile(q, dim=-1))
nutils.add_to_df(head_df, "mover_attn_no_ind", mover_attn_flat[:, ~induction_mask.flatten()].quantile(q, dim=-1))
nutils.add_to_df(head_df, "mover_dla_no_ind", mover_dla_flat[:, ~induction_mask.flatten()].quantile(q, dim=-1))
_ = nutils.add_to_df(head_df, "ratio_dla_no_ind", ratio_dla_flat[:, ~induction_mask.flatten()].quantile(q, dim=-1))
nutils.show_df(head_df.sort_values("mover_dla_no_ind", ascending=False))

# %%
path = f"/workspace/MoverHeads/head_scores/{MODEL_NAME}.csv"
head_df.to_csv(path)
print("Just saved to path:", path)
# %%
