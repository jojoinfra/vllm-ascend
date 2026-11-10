# SPDX-License-Identifier: Apache-2.0
"""Bridge an installed vLLM DSpark speculative step to the V4.1 learned draft.

The target sampler owns verification. The drafter receives the sampler's real
accepted prefix plus its replacement or bonus token, and commits only the target
inputs corresponding to that prefix. Draft self-attention never writes target KV.
"""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

import torch

from vllm_ascend import envs

from .dspark import DSparkDraft
from .spec_config import is_v41_spec


def accepted_target_rows(scheduled_inputs, proposals, sampled):
    """Return committed input count from the core rejection sampler's emitted row."""
    if scheduled_inputs < len(proposals) or len(sampled) > len(proposals) + 1:
        raise ValueError("verification shape is incompatible with the scheduled proposal")
    accepted = max(0, len(sampled) - 1)
    if sampled[:accepted] != proposals[:accepted]:
        raise ValueError("verifier output prefix differs from scheduled proposal")
    if proposals and not sampled:
        raise ValueError("speculative verify without a sampled target token")
    return scheduled_inputs - len(proposals) + accepted, accepted


class RequestSlots:
    """Stable rank-local rows with a logical committed position per live request."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.slots = {}
        self.last = {}

    def assign(self, request_ids):
        if len(request_ids) > self.capacity or len(set(request_ids)) != len(request_ids):
            raise ValueError("invalid active request set")
        for req in self.slots.keys() - set(request_ids):
            del self.slots[req]
            del self.last[req]
        free = iter(sorted(set(range(self.capacity)) - set(self.slots.values())))
        for req in request_ids:
            if req not in self.slots:
                self.slots[req] = next(free)
                self.last[req] = -1

    def commit(self, req, positions, retained, max_context):
        if not positions or positions[0] < 0 or positions[-1] >= max_context:
            raise ValueError("target verifier positions outside context")
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError("non-contiguous target query")
        if positions[0] > self.last[req] + 1 or not 0 <= retained <= len(positions):
            raise ValueError("missing committed target context for draft")
        self.last[req] = positions[0] + retained - 1


def can_propose(last_target_position, prompt_length, block_size, max_context):
    # The seed occupies last+1 and the k draft proposals last+2 through last+k+1.
    return last_target_position >= prompt_length and last_target_position + block_size + 1 < max_context


def _make_proposer_type(native_base):
    class V41Proposer(native_base):
        def __init__(self, vllm_config, device, runner):
            super().__init__(vllm_config, device, runner)
            self.v41_runner = runner
            self.capacity = vllm_config.scheduler_config.max_num_seqs
            self.ledger = RequestSlots(self.capacity)
            self.receipt = None
            self.ready = False
            self.blocks_completed = 0
            self.kv_cache_gid = 0

        def load_model(self, model):
            from vllm.distributed.parallel_state import get_ep_group

            from vllm_ascend.model_loader.deepseek_v41 import DeepseekV41ModelLoader

            from .fused_moe import install_native_layers

            config = self.vllm_config
            if getattr(config.scheduler_config, "async_scheduling", False):
                raise ValueError("V4.1 DSpark currently requires synchronous speculative scheduling")
            target = getattr(model, "backbone", None)
            if target is None:
                target = self.v41_runner.get_model().backbone
            if not hasattr(target, "embed") or not hasattr(target, "head"):
                raise TypeError("expected target backbone with tied embedding and full LM head")
            ep = get_ep_group()
            self.model = DSparkDraft(
                config.model_config.hf_text_config,
                target,
                device=self.device,
                max_requests=self.capacity + 1,
                max_model_len=config.model_config.max_model_len,
                ep_group=ep.device_group,
            )
            root = DeepseekV41ModelLoader(config.load_config)._checkpoint_dir(config.model_config)
            receipts = envs.VLLM_ASCEND_DSV41_RESULTS_DIR
            if receipts:
                directory = Path(receipts)
                directory.mkdir(parents=True, exist_ok=True)
                self.receipt = directory / f"dspark-worker-rank-{ep.rank_in_group}.jsonl"
            self.model.load_weights(
                root,
                receipt=Path(receipts) / f"dspark-weights-rank-{ep.rank_in_group}.json" if receipts else None,
            )
            install_native_layers(self.model, prefix="mtp", receipt_dir=receipts)
            self.model.eval()
            self.ready = True

        def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
            # Capture and backend initialization query the target's only real scheduler
            # cache group. DSpark's projected target-hidden history is owned by DSparkDraft.
            self.kv_cache_gid = 0
            self.block_size = self.vllm_config.cache_config.block_size

        def set_per_group_attn_metadata(self, gid, block_table, slot_mapping):
            # Our draft addresses independent, absolute-position state; no scheduler
            # pages are allocated or written for the draft's projected main KV.
            return None

        def _block(self, seeds, slots, positions, *, role):
            from vllm_ascend.ascend_forward_context import set_ascend_forward_context

            width = self.capacity * self.model.k
            dp = self.vllm_config.parallel_config.data_parallel_size
            across = torch.full((dp,), width, dtype=torch.int32, device="cpu")
            with set_ascend_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                num_tokens=width,
                num_tokens_across_dp=across,
                is_draft_model=True,
                model_instance=self.model,
            ):
                result = self.model(seeds, slots, positions)
            self.blocks_completed += 1
            if self.receipt:
                with self.receipt.open("a") as stream:
                    print(
                        json.dumps(dict(step=self.blocks_completed, role=role, tokens_per_dp_rank=width)),
                        file=stream,
                    )
            return result

        def _inputs(self):
            # An isolated, zero-initialized row handles padding and idle DP ranks.
            # Each rank enters the same three EP collectives with the same fixed shape.
            seeds = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
            return seeds, torch.full_like(seeds, self.capacity), torch.zeros_like(seeds)

        @torch.inference_mode()
        def dummy_run(self, *args, **kwargs):
            if self.ready:
                self._block(*self._inputs(), role="idle")

        @torch.inference_mode()
        def worker_step(self, sampled, scheduler, positions, aux):
            if not self.ready:
                raise RuntimeError("DSpark original weights have not been loaded")
            if isinstance(aux, (list, tuple)):
                aux = torch.cat(aux, dim=-1)
            if aux is None or aux.ndim != 2 or aux.shape[-1] != self.model.mtp[0].main_proj.weight.shape[-1]:
                raise ValueError("missing V4.1 target inputs for the configured auxiliary layers")
            reqs = list(self.v41_runner.input_batch.req_ids)
            self.ledger.assign(reqs)
            if torch.is_tensor(sampled):
                sampled = sampled.tolist()
            if len(sampled) != len(reqs):
                raise ValueError("sampler output disagrees with active requests")
            valid = [[int(token) for token in row if token >= 0] for row in sampled]
            pos = positions.reshape(-1).tolist()
            context = self.vllm_config.model_config.max_model_len
            seeds, slots, last = self._inputs()
            live_rows, commits, commit_slots, events = [], [], [], []
            start = 0
            for row, req in enumerate(reqs):
                count = scheduler.num_scheduled_tokens[req]
                if not count:
                    continue
                query = pos[start : start + count]
                draft = list(scheduler.scheduled_spec_decode_tokens.get(req, ()))
                retained, accepted = accepted_target_rows(count, draft, valid[row])
                self.ledger.commit(req, query, retained, context)
                index = self.ledger.slots[req]
                commits.extend(range(start, start + retained))
                commit_slots.extend([index] * retained)
                end = self.ledger.last[req]
                prompt_len = len(self.v41_runner.requests[req].prompt_token_ids)
                if valid[row] and can_propose(end, prompt_len, self.model.k, context):
                    live_rows.append(row)
                    seeds[row], slots[row], last[row] = valid[row][-1], index, end
                if self.receipt:
                    events.append(
                        dict(
                            row=row,
                            req_id=req,
                            start=query[0],
                            scheduled=count,
                            proposed=draft,
                            accepted=accepted,
                            rejected=len(draft) - accepted,
                            committed_input_count=retained,
                            last_committed=end,
                            emitted=valid[row],
                        )
                    )
                start += count
            if commits:
                ix = torch.tensor(commits, device=self.device, dtype=torch.long)
                self.model.commit_target(
                    aux.index_select(0, ix),
                    torch.tensor(commit_slots, device=self.device),
                    positions.reshape(-1).index_select(0, ix),
                )
            out = self._block(seeds, slots, last, role="sampler")
            tokens = out.token_ids.tolist()
            proposal = [tokens[i] if i in live_rows else [] for i in range(len(reqs))]
            if self.receipt:
                confidences = out.confidence.tolist()
                for event in events:
                    i = event["row"]
                    event["next_proposal"] = proposal[i]
                    event["confidence"] = confidences[i] if proposal[i] else []
                with self.receipt.open("a") as stream:
                    print(json.dumps(events, separators=(",", ":")), file=stream)
            return proposal

    return V41Proposer


def install_spec_runner():
    """Install a model-scoped adapter for the v0.27 A5 model-runner interface."""
    import vllm_ascend.ops  # noqa: F401  (initialize custom ops before the runner)
    from vllm_ascend.spec_decode import AscendDSparkProposer
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner._get_drafter
    if getattr(original, "_handles_v41_dspark", False):
        return
    proposer_type = _make_proposer_type(AscendDSparkProposer)
    propose = NPUModelRunner.propose_draft_token_ids
    uses_aux = NPUModelRunner._eagle3_uses_aux_hidden_state
    fits = NPUModelRunner._input_fits_in_drafter

    @wraps(original)
    def create(self):
        if is_v41_spec(self.speculative_config):
            return proposer_type(self.vllm_config, self.device, self)
        return original(self)

    @wraps(propose)
    def propose_tokens(
        self,
        valid_sampled_token_ids,
        sampling_metadata,
        scheduler_output,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        positions,
        num_scheduled_tokens,
        hidden_states,
        aux_hidden_states=None,
        sample_hidden_states=None,
        target_model_batch_desc=None,
    ):
        if isinstance(self.drafter, proposer_type):
            return self.drafter.worker_step(valid_sampled_token_ids, scheduler_output, positions, aux_hidden_states)
        return propose(
            self,
            valid_sampled_token_ids,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            positions,
            num_scheduled_tokens,
            hidden_states,
            aux_hidden_states,
            sample_hidden_states,
            target_model_batch_desc,
        )

    @wraps(fits)
    def input_fits(self, common_attn_metadata):
        if isinstance(self.drafter, proposer_type):
            # Even a context-tail row must commit its verified hidden states. The
            # complete-block space check belongs to each request in worker_step.
            return common_attn_metadata is not None
        return fits(self, common_attn_metadata)

    @wraps(uses_aux)
    def auxiliary_outputs(self):
        return is_v41_spec(self.speculative_config) or uses_aux(self)

    def complete_draft_collectives(original_step):
        @wraps(original_step)
        def step(self, *args, **kwargs):
            draft = getattr(self, "drafter", None)
            if not isinstance(draft, proposer_type) or not draft.ready:
                return original_step(self, *args, **kwargs)
            before = draft.blocks_completed
            result = original_step(self, *args, **kwargs)
            if before == draft.blocks_completed:
                # A DP dummy batch or context-length fallback can skip the
                # proposer callback; peers may already be entering its MoE.
                draft.dummy_run()
            return result

        return step

    NPUModelRunner._dummy_run = complete_draft_collectives(NPUModelRunner._dummy_run)
    NPUModelRunner.sample_tokens = complete_draft_collectives(NPUModelRunner.sample_tokens)
    create._handles_v41_dspark = True
    NPUModelRunner._input_fits_in_drafter = input_fits
    NPUModelRunner._get_drafter = create
    NPUModelRunner.propose_draft_token_ids = propose_tokens
    NPUModelRunner._eagle3_uses_aux_hidden_state = auxiliary_outputs
