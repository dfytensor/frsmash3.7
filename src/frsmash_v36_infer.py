"""FRSMASH v3.6 独立推理模块 (不影响训练 forward)
backbone 单步用 fused_recurrent_hgrn (逐 token 优化), recall/slowmem state carry.
generate: prefill(forward chunked) → 逐 token(fused_recurrent) + top-k/top-p/rep penalty
generate_cg: CUDA Graph 加速版 (消除 launch overhead)
"""
import torch, torch.nn.functional as F, math
from fla.ops.hgrn import fused_recurrent_hgrn


@torch.no_grad()
def _mfl_step(mfl, x, state):
    b, s, d = x.shape; heads, ns, ds = mfl.heads, mfl.n_slots, mfl.d_sub
    combined = mfl.combined(x).view(b, s, 4, heads, -1)
    out, out1, out2, out3 = combined.unbind(2)
    out = out.permute(0, 3, 1, 2); out1 = out1.permute(0, 3, 1, 2)
    out2 = out2.permute(0, 3, 1, 2); out3 = out3.permute(0, 3, 1, 2)
    sg = mfl.slot_proj(x).reshape(b, s, 4, ns, ds).permute(0, 1, 3, 2, 4)
    af = torch.sigmoid(sg[..., 0, :]); ff = torch.sigmoid(sg[..., 1, :])
    i_f = torch.sigmoid(sg[..., 2, :]); cf = torch.tanh(sg[..., 3, :])
    A = af * ff + (1 - af); B = af * i_f * cf
    A_t = A.permute(0, 2, 1, 3).contiguous(); B_t = B.permute(0, 2, 1, 3).contiguous()
    bns = b * ns
    g = torch.log(A_t.clamp(min=1e-8)).reshape(bns, 1, ds)
    xt = B_t.reshape(bns, 1, ds)
    st_in = state.reshape(bns, ds) if state is not None else None
    H, st = fused_recurrent_hgrn(xt, g, initial_state=st_in, output_final_state=True)
    H = H.reshape(b, ns, 1, ds); new_state = st.reshape(b, ns, ds)
    H_cat = H.permute(0, 2, 1, 3).reshape(b, 1, d)
    out4 = H_cat.reshape(b, 1, heads, mfl.d_head).permute(0, 3, 1, 2)
    cat = torch.cat([out, out1, out2, out3, out4], dim=-1)
    cat_flat = cat.transpose(1, 2).reshape(b, 1, -1)
    return mfl.gen_norm(mfl.gen_gate(cat_flat)), new_state


@torch.no_grad()
def _ssm_layer_step(layer, x, state):
    h = layer.norm1(x)
    ssm_out, s = _mfl_step(layer.ssm, h, state)
    x = x + ssm_out
    x = x + layer.ffn(layer.norm2(x))
    return x, s


@torch.no_grad()
def generate_step(model, token_id, states, h_slow, recall_state, pos):
    dt = model.head.weight.dtype
    x = model.em(token_id).to(dt) + model.pe[pos:pos+1].to(dt)
    h = x; new_states = []
    for i, layer in enumerate(model.layers):
        h, s = _ssm_layer_step(layer, h, states[i] if states else None)
        new_states.append(s)
    x_ash = model.final_norm(h[:, 0])
    inp = model.mem_input_proj(x[:, 0])
    y_slow, h_slow = model.slow_cell.step(inp, h_slow)
    x_mem = model.mem_proj(y_slow)
    o_rec, recall_state = model.recall.step(x[:, 0].float(), recall_state)
    x_recall = model.recall_norm(o_rec.to(dt))
    cat = torch.cat([x_ash, x_mem], dim=-1)
    gate = model.fusion_gate(cat)
    fused = model.fusion_norm(gate * x_ash + (1 - gate) * x_mem + x[:, 0]) + x_recall
    return model.head(fused), new_states, h_slow, recall_state, pos + 1


def _sample(last, ids, top_k, top_p, rep_penalty, temperature):
    lg = last / max(temperature, 1e-4)
    if rep_penalty != 1.0:
        for t in set(ids): lg[t] /= rep_penalty
    if top_k > 0:
        v, _ = torch.topk(lg, min(top_k, lg.size(-1))); lg[lg < v[-1]] = float('-inf')
    if top_p < 1.0:
        probs = torch.softmax(lg, -1); sp, si = torch.sort(probs, descending=True)
        mask = torch.cumsum(sp, -1) > top_p; mask[1:] = mask[:-1].clone(); mask[0] = False
        lg[si[mask]] = float('-inf')
    return int(torch.multinomial(torch.softmax(lg, -1), 1))


@torch.no_grad()
def prefill_chunked(model, prompt_ids, chunk=512):
    """分块 prefill, 三路 state carry, 峰值显存=单块 (只留最后一块 logits, 不拼接)"""
    states = None; hs = None; rs = None; last_lg = None
    for i in range(0, prompt_ids.size(1), chunk):
        c = prompt_ids[:, i:i + chunk]
        last_lg, states, hs, rs = model(c, states=states, h_slow=hs, recall_state=rs, return_state=True, pos_offset=i)
    return last_lg, states, hs, rs


@torch.no_grad()
def generate(model, prompt_ids, max_new=150, top_k=0, top_p=1.0, rep_penalty=1.0,
             temperature=1.0, eos=2, prefill_chunk=512):
    dev = prompt_ids.device
    logits, states, h_slow, recall_state = prefill_chunked(model, prompt_ids, prefill_chunk)
    ids = prompt_ids.tolist()[0]; pos = prompt_ids.size(1)
    last = logits[0, -1]; new_ids = []
    for _ in range(max_new):
        nxt = _sample(last, ids, top_k, top_p, rep_penalty, temperature)
        if eos is not None and nxt == eos: break
        ids.append(nxt); new_ids.append(nxt)
        sl, states, h_slow, recall_state, pos = generate_step(
            model, torch.tensor([[nxt]], device=dev), states, h_slow, recall_state, pos)
        last = sl[0]
    return new_ids


@torch.no_grad()
def generate_cg(model, prompt_ids, max_new=150, top_k=0, top_p=1.0, rep_penalty=1.0,
                temperature=1.0, eos=2, prefill_chunk=512):
    """CUDA Graph accelerated generation (capture whole step, eliminate launch overhead)"""
    dev = prompt_ids.device
    VS = model.head.out_features
    logits, states, h_slow, recall_state = prefill_chunked(model, prompt_ids, prefill_chunk)
    dt = model.head.weight.dtype
    tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
    st_buf = [s.clone() for s in states]
    hs_buf = h_slow.clone()
    rs_buf = recall_state.clone()
    pos_buf = torch.tensor([prompt_ids.size(1)], device=dev, dtype=torch.long)
    out_buf = torch.empty(1, VS, device=dev, dtype=dt)

    def _step():
        x = model.em(tok).to(dt) + model.pe.index_select(0, pos_buf).to(dt)
        h = x; ns = []
        for i, layer in enumerate(model.layers):
            h, s = _ssm_layer_step(layer, h, st_buf[i]); ns.append(s)
        x_ash = model.final_norm(h[:, 0])
        inp = model.mem_input_proj(x[:, 0])
        y, nhs = model.slow_cell.step(inp, hs_buf)
        x_mem = model.mem_proj(y)
        o, nrs = model.recall.step(x[:, 0].float(), rs_buf)
        x_r = model.recall_norm(o.to(dt))
        cat = torch.cat([x_ash, x_mem], -1)
        g = model.fusion_gate(cat)
        fused = model.fusion_norm(g * x_ash + (1 - g) * x_mem + x[:, 0]) + x_r
        out_buf.copy_(model.head(fused))
        for i in range(len(st_buf)): st_buf[i].copy_(ns[i])
        hs_buf.copy_(nhs); rs_buf.copy_(nrs)

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): _step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): _step()

    ids = prompt_ids.tolist()[0]; last = logits[0, -1]; new_ids = []
    for _ in range(max_new):
        nxt = _sample(last, ids, top_k, top_p, rep_penalty, temperature)
        if eos is not None and nxt == eos: break
        ids.append(nxt); new_ids.append(nxt)
        tok.copy_(torch.tensor([[nxt]], device=dev))
        g.replay(); pos_buf += 1
        last = out_buf[0]
    return new_ids
