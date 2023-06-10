# %%
from neel.imports import *
from neel_plotly import *
# %%
import neel.utils as nutils
# %%
model = HookedTransformer.from_pretrained("gpt2-small")
# %%
prompts = ["When Alex and Tim went to the store, Alex gave the bag to",
 "When Alex and Tim when to the store, Tim gave the bag to"]
tokens = model.to_tokens(prompts)
print(tokens)
logits, cache = model.run_with_cache(tokens)
alex_token = model.to_single_token(" Alex")
tim_token = model.to_single_token(" Tim")
# %%
decomposed_resid, labels = cache.get_full_resid_decomposition(expand_neurons=False, pos_slice=-1, return_labels=True)
decomposed_resid = cache.apply_ln_to_stack(decomposed_resid, pos_slice=-1)
print(decomposed_resid.shape)
alex_unembed = model.W_U[:, alex_token]
tim_unembed = model.W_U[:, tim_token]
logit_diff = alex_unembed - tim_unembed
# %%
dla = einops.einsum(decomposed_resid, logit_diff, "component batch d_model, d_model -> component batch")
line(dla.T, x=labels, title="Difference in Logits between Alex and Tim", line_labels=["Should be negative", "Should be positive"])
# %%
SPACE = "·"
NEWLINE="↩"
TAB = "→"
def process_token(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

process_tokens = lambda l: [process_token(s) for s in l]
process_tokens_index = lambda l: [f"{process_token(s)}/{i}" for i,s in enumerate(l)]

# %%
token_list = process_tokens_index(tokens[0])

layer = 9
head = 9
imshow(cache["pattern", layer][:, head], facet_col=0, facet_labels=["Tim", "Alex"], title=f"Head {head} in Layer {layer}", x=token_list, y=token_list)
# %%
tim_decomposed_resid, labels = cache.get_full_resid_decomposition(layer=9, expand_neurons=False, pos_slice=4, return_labels=True)

print(tim_decomposed_resid.shape)

unembed_vec = model.blocks[layer].attn.W_V[head] @ model.blocks[layer].attn.W_O[head] @ (tim_unembed)

line(tim_decomposed_resid[:, 0, :] @ unembed_vec, x=labels, title="IO residual stream -> unembed via L9H9")
line(tim_decomposed_resid.norm(dim=-1).T, title="Norm of Tim Resid Stream")
line((tim_decomposed_resid[:, 0, :] @ unembed_vec)/tim_decomposed_resid.norm(dim=-1)[:, 0], x=labels, title="IO residual stream -> unembed via L9H9 divided by norm")
line(tim_decomposed_resid.norm(dim=-1).T, title="Norm of Tim Resid Stream")
# %%
imshow(cache["pattern", 0][0, :, :5, :5], facet_col=0, x=token_list[:5], y=token_list[:5])
# %%
