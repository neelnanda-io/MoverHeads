# %%
# Setup
from neel.imports import *
from neel_plotly import *

# %%
import neel.utils as nutils

torch.set_grad_enabled(False)
# %%
n_ctx = 256
model = HookedTransformer.from_pretrained("pythia-70m")
dataset = load_dataset("NeelNanda/pile-10k", split="train")
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
# %%
# Utils

SPACE = "·"
NEWLINE = "↩"
TAB = "→"


def process_token(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE + "\n")
    s = s.replace("\t", TAB)
    return s


process_tokens = lambda l: [process_token(s) for s in l]
process_tokens_index = lambda l: [f"{process_token(s)}/{i}" for i, s in enumerate(l)]


def create_vocab_df(logit_vec, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(
            model.to_str_tokens(torch.arange(model.cfg.d_vocab))
        )
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)


from html import escape
import colorsys

from IPython.display import display


def create_html(strings, values, saturation=0.5):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", "<br/>").replace("\t", "&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    # scale values
    max_value = max(max(values), -min(values))
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for s, v in zip(processed_strings, scaled_values):
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))


s = create_html(["a", "b\nd", "c        d"], [1, -2, -3])

def add_to_df(df, name, tensor):
    df[name] = to_numpy(tensor.flatten())
    return df
# %%
# Config
batch_size = 32
tokens = pile_tokens[:batch_size]
logits, cache = model.run_with_cache(tokens)


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
        [tokens[i][argmax_token_index[:, :, i, :]] for i in range(len(tokens))]
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
# Actually running the model on the data distribution
torch.set_grad_enabled(False)
num_prompts = 288
batch_size = 32
head_dla_list = []
mover_attn_score_list = []
mover_dla_score_list = []
plps_list = []

for i in tqdm.tqdm(range(0, num_prompts, batch_size)):
    tokens = pile_tokens[i : i + batch_size]
    logits, cache = model.run_with_cache(tokens)
    plps_list.append(model.loss_fn(logits, tokens, per_token=True))
    head_dla = get_head_dla(cache, tokens)
    head_dla_list.append(head_dla)
    mover_attn_score_list.append(get_mover_attn_score(cache, tokens))
    mover_dla_score_list.append(get_mover_dla_score(cache, tokens))

# %%
plps = torch.cat(plps_list, dim=0)
head_dla = torch.cat(head_dla_list, dim=2)
print(head_dla.shape)
mover_attn_score = torch.cat(mover_attn_score_list, dim=2)
mover_dla_score = torch.cat(mover_dla_score_list, dim=2)

head_dla_flat = einops.rearrange(
    head_dla, "layer head batch pos -> (layer head) (batch pos)"
)
mover_attn_flat = einops.rearrange(
    mover_attn_score, "layer head batch pos -> (layer head) (batch pos)"
)
mover_dla_flat = einops.rearrange(
    mover_dla_score, "layer head batch pos -> (layer head) (batch pos)"
)



# %%
head_df = pd.DataFrame(
    {
        "L": [l for l in range(n_layers) for h in range(n_heads)],
        "H": [h for l in range(n_layers) for h in range(n_heads)],
        "label": model.all_head_labels(),
    }
)

add_to_df(head_df, "head_dla", head_dla_flat.quantile(0.9, dim=-1))
add_to_df(head_df, "mover_attn", mover_attn_flat.quantile(0.9, dim=-1))
add_to_df(head_df, "mover_dla", mover_dla_flat.quantile(0.9, dim=-1))
add_to_df(head_df, "dla_ratio", (mover_dla_flat / head_dla_flat).quantile(0.9, dim=-1))
head_df.style.background_gradient("coolwarm")

# %%
token_subset = pile_tokens[:num_prompts]
num_copies = torch.tril(token_subset[:, :, None] == token_subset[:, None, :], -1).sum(-1)[:, 1:]
is_movable = num_copies > 0
imshow(is_movable[:20], title="Is Movable", xaxis="Pos", yaxis="Prompt")
imshow(num_copies[:20], title="Num Copies", xaxis="Pos", yaxis="Prompt")
print("Frac Movable:", is_movable.float().mean().item())


# %%
# Exploration!
head_labels = model.all_head_labels()
quantiles = torch.tensor(
    [0.0, 1e-3, 1e-2, 5e-2, 1e-1, 0.25, 0.5, 0.75, 9e-1, 95e-2, 99e-2, 999e-3, 1.0]
).cuda()
quantile_labels = [
    "0.0",
    "1e-3",
    "1e-2",
    "5e-2",
    "1e-1",
    "0.25",
    "0.5",
    "0.75",
    "9e-1",
    "95e-2",
    "99e-2",
    "999e-3",
    "1.0",
]

line(
    head_dla_flat.quantile(q=quantiles, dim=-1),
    x=head_labels,
    line_labels=quantile_labels,
    range_y=(-5, 8),
    title="Head DLA",
)
mover_attn_score_flat = einops.rearrange(
    mover_attn_score, "layer head batch pos -> (layer head) (batch pos)"
)
line(
    mover_attn_score_flat.quantile(q=quantiles, dim=-1),
    x=head_labels,
    line_labels=quantile_labels,
    # range_y=(-5, 8),
    title="Mover Attn Scores",
)
mover_dla_score_flat = einops.rearrange(
    mover_dla_score, "layer head batch pos -> (layer head) (batch pos)"
)
line(
    mover_dla_score_flat.quantile(q=quantiles, dim=-1),
    x=head_labels,
    line_labels=quantile_labels,
    range_y=(-5, 8),
    title="Mover dla Scores",
)
line(
    (mover_dla_score_flat/head_dla_flat).quantile(q=quantiles, dim=-1),
    x=head_labels,
    line_labels=quantile_labels,
    range_y=(-5, 8),
    title="Ratio dla Scores",
)
line(
    [mover_dla_score_flat.quantile(0.9, dim=-1)/head_dla_flat.quantile(0.9, dim=-1),
     (mover_dla_score_flat/head_dla_flat).quantile(0.9, -1)],
    x=head_labels,
    range_y=(-5, 8),
    title="Mover DLA / Head DLA",
)
# %%
scatter(x=mover_dla_score_flat.quantile(0.9, dim=-1), y=mover_attn_score_flat.quantile(0.9, dim=-1), color=head_dla_flat.quantile(0.9, dim=-1), hover=head_labels, xaxis="Mover DLA", yaxis="Mover Attn")
scatter(x=mover_dla_score_flat.quantile(0.9, dim=-1), y=head_dla_flat.quantile(0.9, dim=-1), color=mover_attn_score_flat.quantile(0.9, dim=-1), hover=head_labels, xaxis="Mover DLA", yaxis="Head DLA", include_diag=True)
# %%
# Debugging specific inputs:
def decompose_token_index(token_index):
    if not isinstance(token_index, int):
        token_index = token_index.item()
    return token_index//n_ctx, token_index % n_ctx

layer = 5
head = 1
full_token_index = head_dla[5, 1].flatten().argmin()
batch_index, pos = decompose_token_index(full_token_index)

tokens = pile_tokens[batch_index, :pos+1]
print(model.to_string(tokens))
print(tokens[-8:])
print(model.to_str_tokens(tokens[-8:]))
print(model.to_string(tokens[-8:]))

logits, cache = model.run_with_cache(tokens)
print(logits.log_softmax(-1)[0, -5:, 138])

# %%
print(logits.log_softmax(-1)[0, -5:, 138])
# %%
rand_tokens_vocab = torch.tensor([i for i in range(1000, 10000) if "  " not in model.to_string(i)]).cuda()

batch_size = 16
ind_seq_len = 200
random_tokens = rand_tokens_vocab[torch.randint(0, len(rand_tokens_vocab), (batch_size, ind_seq_len))]
bos_tokens = torch.full(
    (batch_size, 1), model.tokenizer.bos_token_id, dtype=torch.long
).cuda()
ind_tokens = torch.cat([bos_tokens, random_tokens, random_tokens], dim=1)
print("ind_tokens.shape", ind_tokens.shape)
_, ind_cache = model.run_with_cache(ind_tokens)

ind_head_scores = einops.reduce(
    ind_cache.stack_activation("pattern").diagonal(ind_seq_len-1, -1, -2),
    "layer batch head diag_pos -> layer head", "mean")
imshow(ind_head_scores, xaxis="Head", yaxis="Layer", title="Induction Head Scores")
add_to_df(head_df, "induction", ind_head_scores)

# %%
px.scatter(head_df, x="induction", y="mover_dla", hover_data=["L", "H"], color="head_dla", trendline="ols", color_continuous_scale="Portland", title="How Induction-y are my found heads?").show()
px.scatter(head_df, x="induction", y="mover_attn", hover_data=["L", "H"], color="mover_dla", color_continuous_scale="Portland", title="How Induction-y are my found heads?").show()
# %%


# %%
# Analysing specific heads
def plot_subsample(x, y, hover=None, num=1000, **kwargs):
    indices = torch.randint(0, len(x), (num,))
    if hover is not None:
        hover = [hover[i.item()] for i in indices]
    return scatter(x[indices], y[indices], hover=hover, **kwargs)
    

traces = []
titles = []
filt_df = head_df[head_df.mover_dla>0.05]
for row in filt_df.iterrows():
    row = row[1]
    layer = row.L
    head = row.H
    label = f"L{layer}H{head}"
    traces.append(plot_subsample(
        x=head_dla[layer, head].flatten(),
        y=mover_dla_score[layer, head].flatten(),
        include_diag=True,
        hover=[f"{p}/{b}" for b in range(num_prompts) for p in range(n_ctx-1)],
        title=f"{label} Mover DLA vs Head DLA",
        xaxis="Head DLA",
        yaxis="Mover DLA",
        opacity=0.5,
        return_fig=True
            ).data[0])
    titles.append(label)
xaxis = "Head DLA"
yaxis = "Mover DLA"
fig = make_subplots(rows=1, cols=len(filt_df), subplot_titles=titles)
fig.update_layout(title="Scatter plot of head DLA vs mover DLA", height=300, xaxis_title="St", **{f"xaxis{x}_title": xaxis for x in range(1, 1+len(filt_df))}, **{f"yaxis{y}_title": yaxis for y in range(1, 2)})
for i, trace in enumerate(traces):
    fig.add_trace(trace, row=1, col=i+1)
fig
# %%
utils.test_prompt("When John and Mary went to the store, John gave the bag to his friend", "Mary", model)
# %%
embed = model.W_E
# line(embed[15])
post_mlp_embed = model.blocks[0].mlp(model.blocks[0].ln2(embed[None])).squeeze(0) + embed

# %%
# copy_score = (post_mlp_embed @ model.OV[3, 3] @ model.W_U).AB
#
# histogram(copy_score.diag())
# histogram(copy_score.max(dim=-1).values)
# histogram(copy_score.min(dim=-1).values)
# %%
# histogram((copy_score[1000:10000] >= copy_score.diag()[1000:10000, None]).float().mean(dim=-1))
# %%
eigenvals_resid = model.OV.eigenvalues
eigenvals_resid_score = ((eigenvals_resid.sum(-1)/eigenvals_resid.abs().sum(-1)).real)
imshow(eigenvals_resid_score)
eigenvals_vocab = (post_mlp_embed @ model.OV @ model.W_U).eigenvalues
eigenvals_vocab_score = ((eigenvals_vocab.sum(-1)/eigenvals_vocab.abs().sum(-1)).real)
imshow(eigenvals_vocab_score)

add_to_df(head_df, "eigenvals_vocab", eigenvals_vocab_score)
add_to_df(head_df, "eigenvals_resid", eigenvals_resid_score)
# %%
factored_full_OV = post_mlp_embed @ model.OV @ model.W_U
def get_head_argmax_value(layer, head, factored_full_OV):
    copy_scores = factored_full_OV[layer, head].AB
    return (torch.arange(d_vocab).cuda() == copy_scores.argmax(dim=-1)).float().mean()
head_df["frac_copy_argmax"] = [get_head_argmax_value(l, h, factored_full_OV).item() for l in range(n_layers) for h in range(n_heads)]
head_df.style.background_gradient("coolwarm")
# %%
# head_df.to_csv("head_df.csv")
def make_induction_mask(tokens):
    equality_check = tokens[:-1, None] == tokens[None, :-1]
    next_equality_check = tokens[1:, None] == tokens[None, 1:]
    return torch.tril(equality_check * next_equality_check, diagonal=-1).any(dim=-1)
make_induction_mask = torch.vmap(make_induction_mask)
induction_mask = make_induction_mask(token_subset)
print(induction_mask.shape, induction_mask.size())
# %%

nutils.show_df(head_df)
# %%
