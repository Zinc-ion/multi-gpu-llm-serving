"""CPU metadata for packed queries against paged KV, including partial blocks."""


def build_prefill_batch(seqs, block_size):
    input_ids, positions, slot_mapping = [], [], []
    cu_seqlens_q, cu_seqlens_k = [0], [0]
    seqlens_q, seqlens_k, logit_indices = [], [], []
    for seq in seqs:
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        if end - start == 1 and end == len(seq):
            # Decode payloads sent to a worker contain only last_token, including
            # when decode shares a packed batch with prefill queries.
            input_ids.append(seq.last_token)
        else:
            input_ids.extend(seq.token_ids[start:end])
        positions.extend(range(start, end))
        seqlens_q.append(end - start)
        seqlens_k.append(end)
        cu_seqlens_q.append(cu_seqlens_q[-1] + end - start)
        cu_seqlens_k.append(cu_seqlens_k[-1] + end)
        if end == len(seq):
            logit_indices.append(cu_seqlens_q[-1] - 1)
        # Warmup runs before the cache exists. At runtime every query has a slot,
        # including chunks starting or ending inside a physical block.
        if seq.block_table:
            slot_mapping.extend(
                seq.block_table[pos // block_size] * block_size + pos % block_size
                for pos in range(start, end)
            )
    max_num_blocks = max(len(seq.block_table) for seq in seqs)
    block_tables = [
        seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
        for seq in seqs
    ] if max_num_blocks else None
    return {
        "input_ids": input_ids, "positions": positions, "slot_mapping": slot_mapping,
        "cu_seqlens_q": cu_seqlens_q, "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max(seqlens_q), "max_seqlen_k": max(seqlens_k),
        "block_tables": block_tables, "logit_indices": logit_indices,
    }
