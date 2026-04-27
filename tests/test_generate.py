"""Tests for VibeBlade TextGenerator."""

import numpy as np

from vibeblade.generate import TextGenerator


class TestSample:
    def test_sample_greedy(self):
        """temperature=0 should always pick argmax."""
        gen = TextGenerator(temperature=0)
        logits = np.array([0.1, -5.0, 3.0, 1.0, 0.5])
        token = gen.sample(logits)
        assert token == 2  # argmax is at index 2

    def test_sample_temperature_shape(self):
        """Output should be a single int."""
        gen = TextGenerator(temperature=0.5, top_k=0, top_p=1.0)
        logits = np.random.randn(100).astype(np.float32)
        token = gen.sample(logits)
        assert isinstance(token, int)
        assert 0 <= token < 100

    def test_softmax_properties(self):
        """Sum to 1, all positive."""
        probs = TextGenerator._softmax(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)

    def test_top_k_limits_choices(self):
        """top_k=3 should only sample from top 3."""
        np.random.seed(42)
        gen = TextGenerator(temperature=1.0, top_k=3, top_p=1.0)
        logits = np.array([10.0, 9.0, 8.0, -100.0, -100.0])
        samples = set()
        for _ in range(200):
            samples.add(gen.sample(logits))
        assert 3 not in samples
        assert 4 not in samples
        assert samples.issubset({0, 1, 2})


class TestGenerate:
    """Tests for the autoregressive generate path."""

    def _make_model_fn(self, vocab_size=50):
        """Create a simple model_fn that returns random logits."""
        np.random.seed(123)
        W = np.random.randn(vocab_size, vocab_size).astype(np.float32) * 0.1

        def model_fn(token_ids):
            one_hot = np.zeros((len(token_ids), vocab_size), dtype=np.float32)
            for i, t in enumerate(token_ids):
                if 0 <= t < vocab_size:
                    one_hot[i, t] = 1.0
            return one_hot @ W
        return model_fn

    def test_generate_returns_array(self):
        model_fn = self._make_model_fn(vocab_size=20)
        gen = TextGenerator(temperature=0)
        result, tps = gen.generate(model_fn, np.array([1, 2, 3]), max_tokens=5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 8  # 3 prompt + 5 generated

    def test_generate_stops_at_eos(self):
        """If model returns EOS (token 2), should stop early."""
        def model_fn_eos(token_ids):
            vocab_size = 20
            logits = np.random.randn(len(token_ids), vocab_size).astype(np.float32)
            logits[-1, :] = -100.0
            logits[-1, 2] = 100.0
            return logits

        gen = TextGenerator(temperature=0)
        result, tps = gen.generate(
            model_fn_eos, np.array([1, 3, 5]), max_tokens=50
        )
        assert isinstance(result, np.ndarray)
        assert len(result) < 53

    def test_generate_calls_callback(self):
        model_fn = self._make_model_fn(vocab_size=20)
        gen = TextGenerator(temperature=0)
        calls = []
        result, tps = gen.generate(
            model_fn, np.array([1, 2, 3]), max_tokens=5,
            on_token=lambda tok, pos: calls.append((tok, pos))
        )
        assert len(calls) == 5

    def test_generate_max_tokens(self):
        model_fn = self._make_model_fn(vocab_size=20)
        gen = TextGenerator(temperature=0)
        result, tps = gen.generate(model_fn, np.array([0]), max_tokens=10)
        assert len(result) <= 11
        assert len(result) >= 2

    def test_generate_with_grammar(self):
        """Grammar mask should constrain token selection."""
        from vibeblade.grammar import GrammarConstraint

        vocab = ['a', 'b', 'c', '1', '2', '3']
        gc = GrammarConstraint.from_regex(vocab, 'abc')
        gen = TextGenerator(temperature=0)

        def model_fn(token_ids):
            # Return logits that would pick '1' without grammar, but grammar blocks it
            logits = np.array([-100.0, -100.0, -100.0, 100.0, -100.0, -100.0],
                              dtype=np.float32)
            return logits.reshape(1, -1)

        result, tps = gen.generate(
            model_fn, np.array([0]), max_tokens=3,
            grammar=gc, vocab=vocab
        )
        # Should never produce '1', '2', or '3' (indices 3, 4, 5)
        for tok in result:
            assert tok < 3, f"Grammar should have blocked token {tok}"
