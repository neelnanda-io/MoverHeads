# %%
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
)
pile_tokens = token_dataset["tokens"].cuda()
print(f"pile_tokens.shape: {pile_tokens.shape}")
# %%
batch_size = 32
tokens = pile_tokens[:batch_size]
logits, cache = model.run_with_cache(tokens)
# %%
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

# %%
W_OU = einops.einsum(
    model.W_O,
    model.W_U,
    "layer head d_head d_model, d_model d_vocab -> layer head d_head d_vocab",
)
print("W_OU.shape:", W_OU.shape)


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


# %%
batch_size = 4
random_tokens = torch.randint(1000, 2000, (batch_size, 128)).cuda()
bos_tokens = torch.full(
    (batch_size, 1), model.tokenizer.bos_token_id, dtype=torch.long
).cuda()
tokens = torch.cat([bos_tokens, random_tokens, random_tokens], dim=1)
logits, cache = model.run_with_cache(tokens)
# %%
line(model.loss_fn(logits, tokens, per_token=True), title="Per Token Correct Log Prob")
# %%
imshow(cache.stack_activation("pattern").diagonal(127, -1, -2).mean(-1).mean(1))
# %%
head_dla = get_head_dla(cache, tokens)
print("head_dla.shape", head_dla.shape)
(
    argmax_with_bos,
    argmax_without_bos,
    max_with_bos,
    max_without_bos,
) = max_prev_attended_to_token(cache)
print("argmax_with_bos.shape", argmax_with_bos.shape)

# %%
imshow(head_dla.clamp(-10, 10)[:, :, :, 128:].mean([-1, -2]))
scatter(
    x=head_dla.clamp(-10, 10).mean([-1, -2]).flatten(),
    y=cache.stack_activation("pattern")
    .diagonal(127, -1, -2)
    .mean(-1)
    .mean(1)
    .flatten(),
    hover=model.all_head_labels(),
)
# %%
line(
    [argmax_with_bos[3, 0, 0], argmax_without_bos[3, 0, 0]],
    title="Argmax of tokens attended to",
)
line(
    [max_with_bos[3, 0, 0], max_without_bos[3, 0, 0]],
    title="max of attention to prev tokens to",
)
# %%
print(tokens.shape)
print(argmax_with_bos.shape)
# predicted_token = torch.stack([tokens[i][argmax_with_bos[:, :, i, :]] for i in range(len(tokens))])
predicted_token = argmax_attn_to_token(argmax_with_bos, tokens)
scatter(
    x=predicted_token[0, 3, 0],
    y=tokens[0, 1:],
    hover=torch.arange(256),
    xaxis="Argmax predicted token",
    yaxis="Actual token",
)

# %%
# Actually running the model on the data distribution
torch.set_grad_enabled(False)
num_prompts = 288
batch_size = 32
head_dla_list = []
attn_index_list = []
for i in tqdm.tqdm(range(0, num_prompts, batch_size)):
    tokens = pile_tokens[i : i + batch_size]
    logits, cache = model.run_with_cache(tokens)
    head_dla = get_head_dla(cache, tokens)
    head_dla_list.append(head_dla)
    (
        argmax_with_bos,
        argmax_without_bos,
        max_with_bos,
        max_without_bos,
    ) = max_prev_attended_to_token(cache)
    attn_index_list.append(
        (argmax_with_bos, argmax_without_bos, max_with_bos, max_without_bos)
    )
head_dla = torch.cat(head_dla_list, dim=2)

print(head_dla.shape)
# %%
layer = 3
head = 0
temp_df = pd.DataFrame(
    {
        "head_dla": to_numpy(head_dla[layer, head].flatten()),
        "batch_index": [f"{b}/{p}" for b in range(num_prompts) for p in range(n_ctx-1)],
        "token": [
            f"{process_token(pile_tokens[b, p])}->{process_token(pile_tokens[b, p+1])}"
            for b in range(num_prompts)
            for p in range(n_ctx-1)
        ],
    }
)
px.histogram(temp_df, x="head_dla", marginal="rug", hover_data=["batch_index", "token"]).show()
# histogram(
#     head_dla[3, 0].flatten(),
#     title="Histogram of head dla for head",
#     xaxis="Head dla",
#     yaxis="Count",
#     marginal="rug",
#     hover_data=[f"{b}/{p}" for b in range(batch_size) for p in range(n_ctx)],
# )
# %%
from html import escape
import colorsys

from IPython.display import display
def create_html(strings, values, saturation=0.5):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [s.replace('\n', '<br/>').replace('\t', '&emsp;').replace(" ", "&nbsp;") for s in escaped_strings]


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
        rgb_color = colorsys.hsv_to_rgb(hue, v, 1) # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = '#%02x%02x%02x' % (int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255))
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))
s = (create_html(["a", "b\nd", "c        d"], [1, -2, -3]))
# %%
prompt_index = 14
position = 79
tokens = pile_tokens[prompt_index]
token_list = process_tokens_index(tokens)
print(model.to_string(tokens))
logits, cache = model.run_with_cache(tokens)
imshow(cache["pattern", layer][0, head], x=token_list, y=token_list, title=f"Attention pattern for head L{layer}H{head}")

create_html(model.to_str_tokens(tokens)[1:], head_dla[layer, head, prompt_index])

# %%
batch_index = prompt_index // batch_size
minibatch_index = prompt_index - batch_index
create_html(model.to_str_tokens(tokens)[1:], attn_index_list[batch_index][3][layer, head, minibatch_index])
# create_html(model.to_str_tokens(tokens)[1:], attn_index_list[batch_index][0][layer, head, minibatch_index])
# %%
def get_mover_attn_score(cache: ActivationCache, tokens: torch.Tensor):
    """
    Return the average attention paid to copies of the next token
    """
    if len(tokens.shape)==1:
        tokens = tokens[None, :]

    patterns = cache.stack_activation("pattern")
    is_next_token = tokens[:, 1:, None] == tokens[:, None, :-1]
    filtered_pattern = patterns[:, :, :, :-1, :-1] * is_next_token[None, :, None, :, :]
    return einops.reduce(filtered_pattern, "layer batch head dest_pos src_pos -> layer head batch dest_pos","sum")
mover_attn_score = get_mover_attn_score(cache, tokens)
l3h0_mover_score = mover_attn_score[layer, head, 0]
scatter(x=l3h0_mover_score, y=head_dla[layer, head, prompt_index], xaxis="Mover Score", yaxis="Head DLA", hover=token_list[1:])
create_html(model.to_str_tokens(tokens)[1:], l3h0_mover_score, 0.4)
create_html(model.to_str_tokens(tokens)[1:], head_dla[layer, head, prompt_index])
# %%
pos = 217
line(cache["pattern", layer][0, head, pos, :pos+1], x=token_list[:pos+1], title=f"Attention pattern of L{layer}H{head} from pos {pos}")
# %%
unembed_vec = model.tokens_to_residual_directions(tokens[217]) / cache["scale"][0, pos, 0]

values = cache["value", layer][0, :pos+1, head, :] # pos, d_head
attns = cache["pattern", layer][0, head, pos, :pos+1] 
line((values) @ model.W_O[layer, head] @ unembed_vec, x=token_list[:pos+1], title=f"DLA via source pos of head (unweighted)")
line((values * attns[:, None]) @ model.W_O[layer, head] @ unembed_vec, x=token_list[:pos+1], title=f"DLA via source pos of head (weighted by attn)")
create_html(model.to_str_tokens(tokens[:pos+1]), (values * attns[:, None]) @ model.W_O[layer, head] @ unembed_vec, 0.4)
create_html(model.to_str_tokens(tokens[:pos+1]), (values) @ model.W_O[layer, head] @ unembed_vec, 0.4)

# %%
src_pos = 109
decomp_resid = cache.