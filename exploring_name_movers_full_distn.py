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
# %%
prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
names = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]
# List of prompts
prompts = []
# List of answers, in the format (correct, incorrect)
answers = []
# List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
answer_tokens = []
for i in range(len(prompt_format)):
    for j in range(2):
        answers.append((names[i][j], names[i][1 - j]))
        answer_tokens.append(
            (
                model.to_single_token(answers[-1][0]),
                model.to_single_token(answers[-1][1]),
            )
        )
        # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
        prompts.append(prompt_format[i].format(answers[-1][1]))
answer_tokens = torch.tensor(answer_tokens).cuda()
print(prompts)
print(answers)
# %%
tokens = model.to_tokens(prompts)

print(list(enumerate(model.to_str_tokens(tokens[0]))))
END = 14
NAME1 = 2
NAME2 = 4
S2 = 10

logits, cache = model.run_with_cache(tokens)
name2_attn = cache.stack_activation("pattern")[:, ::2, :, END, NAME2].mean(1).flatten()
name1_attn = cache.stack_activation("pattern")[:, 1::2, :, END, NAME1].mean(1).flatten()
io_attn = (name1_attn + name2_attn)/2
line([name1_attn, name2_attn, io_attn], x=model.all_head_labels(), line_labels=["name1", "name2", "io"], title="Attn to IO")

stack, labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
unembed_dirs = (model.W_U[:, answer_tokens[:, 0]] - model.W_U[:, answer_tokens[:, 1]]).T
dla = (unembed_dirs * stack).sum([-2, -1])
line(dla, x=labels, title="LDA")

scatter(x=io_attn, y=dla[:144], hover=model.all_head_labels(), color=[l for l in range(12) for h in range(12)], xaxis="Attn", yaxis="DLA", title="Attn vs DLA")
# %%
name_mover_layers = np.array([9, 9, 9, 9, 10, 10, 11])
name_mover_heads = np.array([6, 9, 8, 0, 0, 7, 10])

head_mask = torch.zeros((n_layers, n_heads)).bool().cuda()
head_mask[name_mover_layers, name_mover_heads] = True
# %%

def get_name_mover_attn_score_old(cache, name_mover_layers=name_mover_layers, name_mover_heads=name_mover_heads):
    patterns = []
    for head, layer in zip(name_mover_heads, name_mover_layers):  
        patterns.append(cache["pattern", layer][:, head, :, 1:]) # 1: to exclude the BOS token
    patterns = torch.stack(patterns, dim=0)
    # return patterns
    patterns = einops.reduce(patterns, "head batch dest_pos src_pos -> batch dest_pos src_pos", "min")
    patterns_max, patterns_argmax = patterns.max(-1)
    return patterns_max, patterns_argmax

def get_name_mover_attn_score(cache, head_mask=head_mask):
    patterns = cache.stack_activation("pattern")[:, :, :, :, 1:]
    patterns = einops.rearrange(patterns, "layer batch head dest_pos src_pos -> batch (layer head) dest_pos src_pos")
    mover_patterns = patterns[:, [9*12+9, 9*12+6], :, :].min(1).values
    # mover_patterns = patterns[:, head_mask.flatten(), :, :].min(1).values
    non_mover_patterns = patterns[:, ~head_mask.flatten(), :, :].max(1).values
    diff_patterns = mover_patterns - non_mover_patterns
    diff_patterns_max, diff_patterns_argmax = diff_patterns.max(-1)
    mover_patterns_val = mover_patterns.gather(-1, diff_patterns_argmax.unsqueeze(-1)).squeeze(-1)
    non_mover_patterns_val = non_mover_patterns.gather(-1, diff_patterns_argmax.unsqueeze(-1)).squeeze(-1)
    return mover_patterns_val, non_mover_patterns_val, diff_patterns_max, diff_patterns_argmax


# line(get_name_mover_attn_score(cache)[0], x=nutils.process_tokens_index(tokens[0], model=model))
mover_patterns_val, non_mover_patterns_val, diff_patterns_max, diff_patterns_argmax = get_name_mover_attn_score(cache)
line(diff_patterns_max)
# %%
W_OU = []
for head, layer in zip(name_mover_heads, name_mover_layers):
    W_OU.append(model.W_O[layer, head] @ model.W_U)
W_OU = torch.stack(W_OU, dim=0)

def get_name_mover_dla_score(cache, name_mover_layers=name_mover_layers, name_mover_heads=name_mover_heads, W_OU=W_OU):
    z = []
    for head, layer in zip(name_mover_heads, name_mover_layers):
        z.append(cache["z", layer][:, :, head, :])
    z = torch.stack(z, dim=0)
    z_dla = z @ W_OU[:, None, :, :]
    z_dla = einops.reduce(z_dla, "head batch pos d_vocab -> batch pos d_vocab", "min")
    z_dla_max, z_dla_argmax = z_dla.max(-1)
    return z_dla_max, z_dla_argmax
z_dla_max, z_dla_argmax = get_name_mover_dla_score(cache)
# imshow(patterns[:, :, 9, :], facet_col=0)
# imshow(patterns[:, :, 14, :], facet_col=0)
# %%
scatter(x=get_name_mover_attn_score(cache).flatten(), y=z_dla_max.flatten(), hover=[f"{b}/{p}/{model.to_string(tokens[b, p])}" for b in range(tokens.shape[0]) for p in range(tokens.shape[1])], color=[p for b in range(tokens.shape[0]) for p in range(tokens.shape[1])])
# %%
torch.set_grad_enabled(False)
name_mover_attn_score_list = []
name_mover_attn_argmax_list = []
name_mover_dla_score_list = []
name_mover_dla_argmax_list = []
for i in tqdm.tqdm(range(0, num_prompts, batch_size)):
    tokens = pile_tokens[i : i + batch_size]
    logits, cache = model.run_with_cache(tokens)
    attn_score, attn_argmax = (get_name_mover_attn_score(cache))
    name_mover_attn_score_list.append(attn_score)
    name_mover_attn_argmax_list.append(attn_argmax)

    dla_score, dla_argmax = get_name_mover_dla_score(cache)
    name_mover_dla_score_list.append(dla_score)
    name_mover_dla_argmax_list.append(dla_argmax)

name_mover_attn_score = torch.cat(name_mover_attn_score_list, dim=0)
name_mover_attn_argmax = torch.cat(name_mover_attn_argmax_list, dim=0)
name_mover_dla_score = torch.cat(name_mover_dla_score_list, dim=0)
name_mover_dla_argmax = torch.cat(name_mover_dla_argmax_list, dim=0)
# %%
token_df = pd.DataFrame({
    "pos": [p for b in range(num_prompts) for p in range(n_ctx)],
    "batch": [b for b in range(num_prompts) for p in range(n_ctx)],
    "attn": to_numpy(name_mover_attn_score.flatten()),
    "attn_argmax": to_numpy(name_mover_attn_argmax.flatten()),
    "dla": to_numpy(name_mover_dla_score.flatten()),
    "dla_argmax": to_numpy(name_mover_dla_argmax.flatten()),
})
token_df["dla_token"] = nutils.process_tokens(name_mover_dla_argmax.flatten(), model=model)

# %%
def make_induction_mask(tokens, device=DEVICE):
    tokens = tokens.to(device)
    equality_check = tokens[:-1, None] == tokens[None, :-1]
    next_equality_check = tokens[1:, None] == tokens[None, 1:]
    return torch.tril(equality_check * next_equality_check, diagonal=-1).any(dim=-1)
make_induction_mask = torch.vmap(make_induction_mask)
induction_mask = make_induction_mask(pile_tokens[:num_prompts])
print(induction_mask.shape, induction_mask.size())
token_df = token_df.query(f"pos<{n_ctx-1}")
token_df["induction"] = to_numpy(induction_mask.flatten())
# %%
# %%
nutils.show_df(token_df.sort_values("attn", ascending=False).head(500))
nutils.show_df(token_df.query("~induction").sort_values("attn", ascending=False).head(500))
# %%
head_token_df = token_df.query("~induction").sort_values("attn", ascending=False).head(500)
for i in range(10):
    batch = head_token_df.batch.iloc[i]
    pos = head_token_df.pos.iloc[i]
    src_pos = head_token_df.attn_argmax.iloc[i]+1
    print(batch, pos, src_pos)
    tokens = pile_tokens[batch, :pos+1]
    logits, cache = model.run_with_cache(tokens)
    for head,layer in zip(name_mover_heads, name_mover_layers):
        nutils.create_html(model.to_str_tokens(tokens), cache["pattern", layer][0, head, -1, :])
    line(cache.stack_activation("pattern")[:, 0, :, -1, src_pos].flatten(), x=model.all_head_labels())
# %%
# %%



# %%
def get_l9h9_minus_l9h6_max(cache):
    return (cache["pattern", 9][:, 9, :, :] - cache["pattern", 9][:, 6, :, :]).abs().sum(-1)

torch.set_grad_enabled(False)
name_mover_attn_score_out_list = []
l9h9_minus_l9h6_max_list = []
for i in tqdm.tqdm(range(0, num_prompts, batch_size)):
    tokens = pile_tokens[i : i + batch_size]
    logits, cache = model.run_with_cache(tokens)
    name_mover_attn_score_out_list.append(get_name_mover_attn_score(cache))
    l9h9_minus_l9h6_max_list.append(get_l9h9_minus_l9h6_max(cache))
#     attn_score, attn_argmax = (get_name_mover_attn_score(cache))
#     name_mover_attn_score_list.append(attn_score)
#     name_mover_attn_argmax_list.append(attn_argmax)

#     dla_score, dla_argmax = get_name_mover_dla_score(cache)
#     name_mover_dla_score_list.append(dla_score)
#     name_mover_dla_argmax_list.append(dla_argmax)

# name_mover_attn_score = torch.cat(name_mover_attn_score_list, dim=0)
# name_mover_attn_argmax = torch.cat(name_mover_attn_argmax_list, dim=0)
# name_mover_dla_score = torch.cat(name_mover_dla_score_list, dim=0)
# name_mover_dla_argmax = torch.cat(name_mover_dla_argmax_list, dim=0)

diff_name_mover_attn_score = torch.cat([i[2] for i in name_mover_attn_score_out_list], dim=0)
l9h9_minus_l9h6_max = torch.cat(l9h9_minus_l9h6_max_list, dim=0)
# %%
token_df = pd.DataFrame({
    "pos": [p for b in range(num_prompts) for p in range(n_ctx)],
    "batch": [b for b in range(num_prompts) for p in range(n_ctx)],
    "attn": to_numpy(diff_name_mover_attn_score.flatten()),
    "l9h9_minus_l9h6_max": to_numpy(l9h9_minus_l9h6_max.flatten()),
    # "attn_argmax": to_numpy(name_mover_attn_argmax.flatten()),
    # "dla": to_numpy(name_mover_dla_score.flatten()),
    # "dla_argmax": to_numpy(name_mover_dla_argmax.flatten()),
})
# token_df["dla_token"] = nutils.process_tokens(name_mover_dla_argmax.flatten(), model=model)
# %%
nutils.show_df(token_df.sort_values("attn", ascending=False).head(500))
# %%
head_token_df = token_df.sort_values("l9h9_minus_l9h6_max", ascending=False).head(500)
for i in range(10):
    batch = head_token_df.batch.iloc[i]
    pos = head_token_df.pos.iloc[i]
    # src_pos = head_token_df.attn_argmax.iloc[i]+1
    tokens = pile_tokens[batch, :pos+1]
    logits, cache = model.run_with_cache(tokens)
    src_pos = cache["pattern", 9][0, 6, -1, :].argmax()
    print(batch, pos, src_pos)
    
    nutils.create_html(model.to_str_tokens(tokens), cache["pattern", 9][0, 9, -1, :])
    nutils.create_html(model.to_str_tokens(tokens), cache["pattern", 9][0, 6, -1, :])
    line(cache.stack_activation("pattern")[:, 0, :, -1, src_pos].flatten(), x=model.all_head_labels())

# %%
