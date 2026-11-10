# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.dspark import DSparkDraft
from vllm_ascend.models.deepseek_v41.spec_runner import RequestSlots, accepted_target_rows, can_propose


def test_rejection_commits_only_target_prefix_and_reuses_slot():
    positions = RequestSlots(2)
    positions.assign(["a", "b"])
    positions.commit("a", list(range(8)), 8, 14)
    positions.commit("b", [0, 1], 2, 14)
    retained, accepted = accepted_target_rows(4, [9, 7, 3], [9, 4])
    assert (retained, accepted) == (2, 1)
    positions.commit("a", [8, 9, 10, 11], retained, 14)
    assert positions.last == {"a": 9, "b": 1}
    assert accepted_target_rows(3, [5, 4], [5, 4, 6]) == (3, 2)
    positions.commit("a", [10], 1, 14)
    slot = positions.slots["a"]
    positions.assign(["b", "c"])
    assert positions.slots["c"] == slot and positions.last["c"] == -1
    with pytest.raises(ValueError):
        positions.commit("c", [10], 1, 14)
    with pytest.raises(ValueError):
        positions.commit("b", [-1, 0], 2, 14)
    with pytest.raises(ValueError):
        positions.commit("b", [1, 3], 2, 14)
    with pytest.raises(ValueError):
        accepted_target_rows(3, [5, 4], [9, 4])
    with pytest.raises(ValueError):
        accepted_target_rows(1, [5, 4], [5])


def test_prefill_and_context_tail_fall_back_per_request():
    assert not can_propose(3, 4, 5, 16)
    assert can_propose(9, 4, 5, 16)
    assert not can_propose(10, 4, 5, 16)
    model = DSparkDraft.__new__(DSparkDraft)
    nn.Module.__init__(model)
    model.max_model_len, model.max_requests, model.vocab_size, model.k = 16, 2, 10, 5
    for seed, slot, position in [(0, -1, 2), (0, 2, 2), (0, 0, -1), (0, 0, 10), (10, 0, 2)]:
        with pytest.raises(ValueError):
            model.propose(torch.tensor([seed]), torch.tensor([slot]), torch.tensor([position]))
    with pytest.raises(ValueError):
        model.commit_target(torch.zeros(1, 4), torch.tensor([0]), torch.tensor([16]))


def test_worker_keeps_only_verified_rows_when_two_requests_are_reordered():
    from vllm_ascend.models.deepseek_v41.spec_runner import _make_proposer_type

    class ToyDraft:
        k = 2
        mtp = [SimpleNamespace(main_proj=SimpleNamespace(weight=torch.empty(3, 4)))]

        def commit_target(self, aux, slots, pos):
            self.last_commit = list(zip(slots.tolist(), pos.tolist(), aux[:, 0].tolist()))

    cls = _make_proposer_type(object)
    proposer = cls.__new__(cls)
    proposer.ready, proposer.capacity, proposer.device, proposer.receipt = True, 2, "cpu", None
    proposer.ledger, proposer.model = RequestSlots(2), ToyDraft()
    states = {k: SimpleNamespace(prompt_token_ids=[0]) for k in ("a", "b")}
    batch = SimpleNamespace(req_ids=["a", "b"])
    proposer.v41_runner = SimpleNamespace(input_batch=batch, requests=states)
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=16))
    proposer._block = lambda seeds, slots, pos, role: SimpleNamespace(
        token_ids=torch.stack((seeds + 1, seeds + 2), dim=1)
    )

    def step(rows, counts, drafts, positions, values):
        return proposer.worker_step(
            rows,
            SimpleNamespace(num_scheduled_tokens=counts, scheduled_spec_decode_tokens=drafts),
            torch.tensor(positions),
            torch.tensor(values)[:, None].expand(-1, 4),
        )

    assert step([[3], [8]], {"a": 1, "b": 1}, {}, [0, 0], [4.0, 5.0]) == [[], []]
    assert step([[4], [9]], {"a": 1, "b": 1}, {}, [1, 1], [6.0, 7.0]) == [[5, 6], [10, 11]]
    batch.req_ids = ["b", "a"]
    assert step(
        [[10, 11, 1], [5, 7]],
        {"b": 3, "a": 3},
        {"b": [10, 11], "a": [5, 6]},
        [2, 3, 4, 2, 3, 4],
        [8.0, 9.0, 10.0, 11.0, 12.0, float("nan")],
    ) == [[2, 3], [8, 9]]
    assert proposer.model.last_commit == [(1, 2, 8.0), (1, 3, 9.0), (1, 4, 10.0), (0, 2, 11.0), (0, 3, 12.0)]
    assert proposer.ledger.last == {"a": 3, "b": 4}
