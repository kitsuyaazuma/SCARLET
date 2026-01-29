import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from scarlet.algorithm.common import CommonServerArgs
from scarlet.algorithm.scarlet import (
    CacheSignal,
    LocalCacheEntry,
    SCARLETClientConfig,
    SCARLETClientTrainer,
    SCARLETDownlinkPackage,
    SCARLETServerHandler,
    SCARLETUplinkPackage,
)

from .test_common import MockDatasetProvider, SimpleModel


@pytest.fixture
def scarlet_server_handler() -> SCARLETServerHandler:
    model = SimpleModel()
    dataset = MockDatasetProvider()
    args = CommonServerArgs(
        dataset=dataset,
        global_round=5,
        num_clients=2,
        sample_ratio=1.0,
        device="cpu",
        kd_epochs=1,
        kd_batch_size=2,
        kd_lr=0.01,
        public_size_per_round=2,
        seed=42,
    )
    # cache_duration: 2 rounds
    return SCARLETServerHandler.from_args(
        args, model, enhanced_era_exponent=2.0, cache_duration=2
    )


def test_scarlet_server_handler_global_update(
    scarlet_server_handler: SCARLETServerHandler,
) -> None:
    handler = scarlet_server_handler

    # --- Round 0: Initial caching ---
    indices_r0 = handler.get_next_indices()
    target_idx = int(indices_r0[0].item())
    soft_labels_r0 = torch.randn(len(indices_r0), 2)
    buffer_r0 = [
        SCARLETUplinkPackage(
            cid=0, soft_labels=soft_labels_r0, indices=indices_r0, metrics={}
        )
    ]
    handler.global_update(buffer_r0)

    # Verify NEWLY_CACHED signals and cache population
    assert handler.cache_signals is not None
    assert torch.all(handler.cache_signals == CacheSignal.NEWLY_CACHED)
    assert handler.global_cache[target_idx].soft_label is not None

    # --- Round 1: Cache Hit ---
    handler.round = 1
    indices_r1 = handler.get_next_indices()

    # Verify target_idx is filtered out from request because it's validly cached
    assert target_idx not in indices_r1.tolist()

    # --- Round 2: Cache Expiry ---
    # Duration is 2, so round 0 labels expire at round 2 or later
    handler.round = 3

    # Request the same target_idx again after expiry
    buffer_r2 = [
        SCARLETUplinkPackage(
            cid=0,
            soft_labels=torch.randn(1, 2),
            indices=torch.tensor([target_idx]),
            metrics={},
        )
    ]
    handler.global_update(buffer_r2)

    # Verify EXPIRED signal is triggered for the stale cache entry
    assert handler.cache_signals is not None
    assert CacheSignal.EXPIRED in handler.cache_signals.tolist()


def test_scarlet_server_handler_enhanced_era() -> None:
    def get_entropy(p: torch.Tensor) -> torch.Tensor:
        """Helper to calculate Shannon entropy"""
        return -torch.sum(p * torch.log(p + 1e-9), dim=-1)

    # High entropy input (nearly uniform)
    p_high = torch.tensor([0.45, 0.55])
    # Low entropy input (already sharp)
    p_low = torch.tensor([0.1, 0.9])

    # 1. Identity Property: beta = 1 should preserve the original distribution
    assert torch.allclose(SCARLETServerHandler.enhanced_era(p_high, 1.0), p_high)
    assert torch.allclose(SCARLETServerHandler.enhanced_era(p_low, 1.0), p_low)

    # 2. Sharpening Property: beta > 1 should reduce entropy
    p_high_sharp = SCARLETServerHandler.enhanced_era(p_high, 2.0)
    p_low_sharp = SCARLETServerHandler.enhanced_era(p_low, 2.0)

    assert get_entropy(p_high_sharp) < get_entropy(p_high)
    assert get_entropy(p_low_sharp) < get_entropy(p_low)

    # 3. Predictable Control: Larger beta results in lower entropy
    p_very_sharp = SCARLETServerHandler.enhanced_era(p_high, 5.0)
    assert get_entropy(p_very_sharp) < get_entropy(p_high_sharp)


def test_scarlet_client_trainer_update_local_cache() -> None:
    # Setup labels and signals
    # Index 10: NEWLY_CACHED, Index 11: CACHED, Index 12: EXPIRED
    global_soft_labels = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
    global_indices = torch.tensor([10, 11, 12])

    # Initialize local cache with a pre-existing label for index 11
    existing_label = torch.tensor([0.5, 0.5])
    local_cache = [LocalCacheEntry(None) for _ in range(20)]
    local_cache[11] = LocalCacheEntry(soft_label=existing_label)
    local_cache[12] = LocalCacheEntry(
        soft_label=torch.tensor([0.9, 0.1])
    )  # Old label to be expired

    cache_signals = torch.tensor(
        [CacheSignal.NEWLY_CACHED, CacheSignal.CACHED, CacheSignal.EXPIRED]
    )

    new_cache, restored_labels = SCARLETClientTrainer.update_local_cache(
        global_soft_labels, global_indices, local_cache, cache_signals
    )

    # 1. Verify NEWLY_CACHED: Should update cache and return the new label
    assert new_cache[10].soft_label is not None
    assert torch.allclose(new_cache[10].soft_label, global_soft_labels[0])
    assert torch.allclose(restored_labels[0], global_soft_labels[0])

    # 2. Verify CACHED: Should reuse existing label from cache
    assert new_cache[11].soft_label is not None
    assert torch.allclose(new_cache[11].soft_label, existing_label)
    assert torch.allclose(restored_labels[1], existing_label)

    # 3. Verify EXPIRED: Should clear cache entry and return the new label from server
    assert new_cache[12].soft_label is None
    assert torch.allclose(restored_labels[2], global_soft_labels[1])

    assert len(restored_labels) == 3


def test_scarlet_client_trainer_worker(tmp_path: Path) -> None:
    model_name = "resnet18"
    dataset = MockDatasetProvider()

    class MockModelSelector:
        def select_model(self, name: Any) -> torch.nn.Module:
            _ = name
            return SimpleModel()

    config = SCARLETClientConfig(
        model_selector=MockModelSelector(),  # type: ignore[arg-type]
        model_name=model_name,  # type: ignore[arg-type]
        dataset=dataset,
        epochs=1,
        batch_size=2,
        lr=0.01,
        kd_epochs=1,
        kd_batch_size=2,
        kd_lr=0.01,
        cid=0,
        seed=42,
        state_path=tmp_path / "scarlet_0.pt",
    )

    # Payload with cache signals
    payload = SCARLETDownlinkPackage(
        soft_labels=torch.randn(1, 2),
        indices=torch.tensor([5]),
        next_indices=torch.tensor([6]),
        cache_signals=torch.tensor([CacheSignal.NEWLY_CACHED]),
    )

    shm_buffer = SCARLETUplinkPackage(
        cid=-1,
        soft_labels=torch.zeros(1, 2),
        indices=torch.zeros(1, dtype=torch.long),
        metrics={},
    )

    stop_event = threading.Event()
    package = SCARLETClientTrainer.worker(
        config, payload, "cpu", stop_event, shm_buffer=shm_buffer
    )

    assert package.cid == 0
    assert config.state_path.exists()
