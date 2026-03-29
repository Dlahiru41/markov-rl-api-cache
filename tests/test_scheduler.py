"""Tests for scheduler job registration and status exposure."""

from src.scheduler.training_scheduler import TrainingScheduler


class DummyMarkov:
    def update(self, *_args, **_kwargs):
        return self


class DummyBuffer:
    def __len__(self):
        return 0


class DummyDQN:
    epsilon = 0.1
    last_loss = 0.0
    buffer = DummyBuffer()

    def train_step(self):
        return None

    def save(self, _path):
        return None

    def load(self, _path):
        return None


class DummyCollector:
    pass


def test_scheduler_registers_all_jobs():
    sch = TrainingScheduler(markov_chain=DummyMarkov(), dqn_agent=DummyDQN(), collector=DummyCollector(), redis_client=None)
    status = sch.get_status()
    assert set(status.keys()) == {"markov_update", "dqn_training", "model_evaluation", "data_cleanup"}
