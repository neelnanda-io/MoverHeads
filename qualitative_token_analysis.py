# %%
from produce_token_metric_df import *
import tqdm
# %%
# Key Variables
"""
induction_mask
is_movable
head_dla_flat
mover_attn_flat
mover_dla_flat
ratio_dla_flat
"""
print("1")
str_tokens = [model.to_str_tokens(pile_tokens[i, :]) for i in range(num_prompts)]
print("2")
str_tokens = [[nutils.process_token(t, model) for t in tokens] for tokens in str_tokens]
print("3")
context_dest_tokens = ["|".join([str_tokens[b][p-i] for i in [3, 2, 1, 0, -1]]) for b in range(num_prompts) for p in range(n_ctx-1)]
print("4")
# %%

# histogram(head_dla[10, 7].flatten(), histnorm="percent", title="L10H7 head_dla")
# histogram(head_dla[11, 10].flatten(), histnorm="percent", title="L11H10 head_dla")
# histogram(mover_dla[11, 10].flatten(), histnorm="percent", title="L11H10 mover_head_dla")
# # %%
# histogram(mover_dla[10, 7].flatten()[is_movable.cpu().flatten()], histnorm="percent", title="L10H7 mover_head_dla")
# %%
# I want to do an investigation of a specific token
# Give in batch, pos, layer, head
# Print plp, head DLA, mover DLA, mover attn, ratio DLA, is induction, is movable
# Print destination token(s)
# Print source token(s)
# Take argmax of mover attn and take previous thing, give the attn to it and DLA from it, and surrounding context. 

# argmax_without_bos_list = []
# max_without_bos_list = []
# predicted_token_list = []
# for i in tqdm.tqdm(range(0, num_prompts, batch_size)):
#     tokens = pile_tokens[i : i + batch_size]
#     logits, cache = model.run_with_cache(tokens)
    
#     _, argmax_without_bos, _, max_without_bos = max_prev_attended_to_token(cache)
#     argmax_without_bos_list.append(argmax_without_bos.to(DEVICE))
#     max_without_bos_list.append(max_without_bos.to(DEVICE))
#     predicted_token = argmax_attn_to_token(argmax_without_bos, tokens)
#     predicted_token_list.append(predicted_token.to(DEVICE))

# argmax_without_bos = torch.cat(argmax_without_bos_list, dim=2)
# print("{argmax_without_bos.shape=}")
# max_without_bos = torch.cat(max_without_bos_list, dim=2)
# predicted_token_list = torch.cat(predicted_token_list, dim=2)



layer = 9
head = 11
token_df = pd.DataFrame({
    "batch":[b for b in range(num_prompts) for p in range(n_ctx-1)],
    "pos":[p for b in range(num_prompts) for p in range(n_ctx-1)],
    
    "plp": to_numpy(plps[:].flatten()),
    "is_induction": to_numpy(induction_mask[:].flatten()),
    "is_movable": to_numpy(is_movable[:].flatten()),

    "head_dla": to_numpy(head_dla[layer, head].flatten()),
    "mover_dla": to_numpy(mover_dla[layer, head].flatten()),
    "mover_attn": to_numpy(mover_attn[layer, head].flatten()),
    "ratio_dla": to_numpy(ratio_dla[layer, head].flatten()),
    "argmax_without_bos": to_numpy(argmax_without_bos[layer, head].flatten()),
    "max_without_bos": to_numpy(max_without_bos[layer, head].flatten()),
    "predicted_token": to_numpy(predicted_token[layer, head].flatten()),
    "tokens": to_numpy(pile_tokens[:num_prompts, :-1].flatten()),
})
token_df["context_dest_tokens"] = context_dest_tokens
# for k, v in token_df.items():
#     if not isinstance(v, list):
#         print(k, v.shape)

# %%
token_df.describe()

token_df["token_curr"] = nutils.process_tokens(to_numpy(pile_tokens[:num_prompts, :-1].flatten()), model)
token_df["token_next"] = nutils.process_tokens(to_numpy(pile_tokens[:num_prompts, 1:].flatten()), model)
# %%
token_df["is_moving"] = (token_df["predicted_token"].values == to_numpy(pile_tokens[:num_prompts, 1:].flatten()))
# %%
def argmax_attn_index_to_context(attn_index, tokens, k=5):
    """
    Given the argmax of the attention, return the context of the attention.
    """
    # attn_index has shape [n_ctx-1]
    # tokens has shape [n_ctx-1]
    # I want to return the string values of the 5 tokens before the earlier context
    context_tokens = []
    for r in range(n_ctx-1):
        context_tokens.append("|".join(nutils.process_tokens(tokens[(attn_index[r]-torch.arange(k-1, -1, -1)).clamp(0, 100000)], model)))
    return context_tokens
context_src_tokens = [argmax_attn_index_to_context(argmax_without_bos[layer, head, i], pile_tokens[i, :-1]) for i in range(num_prompts)]
# %%
token_df["context_src_tokens"] = [context_src_tokens[i][j] for i in range(num_prompts) for j in range(n_ctx-1)]

# %%
token_df.head()
# %%
token_df_no_ind = token_df[~token_df["is_induction"]]
# %%
nutils.show_df(token_df_no_ind.sort_values("mover_dla", ascending=False).head(100))
# %%
# def argmax_attn_index_to_context(attn_index, tokens, k=5):
#     """
#     Given the argmax of the attention, return the context of the attention.
#     """
#     # attn_index has shape [n_ctx-1]
#     # tokens has shape [n_ctx-1]
#     # I want to return the string values of the 5 tokens before the earlier context
#     context_tokens = []
#     for r in range(n_ctx-1):
#         context_tokens.append("|".join(nutils.process_tokens(tokens[(attn_index[r]-torch.arange(k-1, -1, -1)).clamp(0, 100000)], model)))
#     return context_tokens
# context_src_tokens = [argmax_attn_index_to_co


# %%
# %%
nutils.show_df(token_df_no_ind.sort_values("mover_dla", ascending=False).head(100))
# %%

batch = 585
p = 129

tokens = pile_tokens[batch, :p+2]
logits, cache = model.run_with_cache(tokens)
nutils.create_html(model.to_str_tokens(tokens), to_numpy(cache["pattern", layer][0, head, p, :p+1]), allow_different_length=True)

resid_decomp, resid_labels = cache.get_full_resid_decomposition(apply_ln=True, expand_neurons=False, pos_slice=-2, return_labels=True)
line(resid_decomp @ model.W_U[:, tokens[-1]], x=resid_labels, title=f"DLA for final token of B={batch},P={p}")
# %%
scatter(x=head_dla[9, 9, :100].flatten(), y=head_dla[10, 7, :100].flatten(), xaxis="L9H9", yaxis="L9H6", title="Head DLA", opacity=0.3)
scatter(x=mover_dla[9, 9, :100].flatten(), y=mover_dla[10, 7, :100].flatten(), xaxis="L9H9", yaxis="L9H6", title="Mover Head DLA", opacity=0.3)
# %%
