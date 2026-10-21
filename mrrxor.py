"""The Mixing Random-Random-XOR (MRRXOR) process: sampling and exact Bayesian beliefs.

MRRXOR is a five-state hidden Markov model (states A-E) that emits two independent random bits and then their XOR,
and then starts over:

    A --0--> B,  A --1--> C           (first random bit, probability 1/2 each)
    B --0--> E,  B --1--> D           (second random bit, probability 1/2 each)
    C --0--> D,  C --1--> E
    D --1--> A,  E --0--> A           (the XOR of the two bits)

In addition, before each emission the hidden state jumps to one of the other four states, chosen uniformly, with
probability eps. Without mixing (eps = 0) this is the RRXOR process of Crutchfield and Feldman (2003).

Edge-emitting convention: T[x, s, t] is the probability of emitting token x and moving to state t from state s.
"""
import torch

A, B, C, D, E = range(5)


def transition_tensor(eps: float = 0.05) -> torch.Tensor:
    """T[x, s, t] = P(emit x, next state t | state s), with the random jump applied before emitting."""
    emit = torch.zeros(2, 5, 5, dtype=torch.float64)
    emit[0, A, B] = emit[1, A, C] = 0.5
    emit[0, B, E] = emit[1, B, D] = 0.5
    emit[0, C, D] = emit[1, C, E] = 0.5
    emit[1, D, A] = emit[0, E, A] = 1.0
    jump = torch.full((5, 5), eps / 4, dtype=torch.float64)
    jump.fill_diagonal_(1 - eps)
    return jump @ emit


def stationary_distribution(T: torch.Tensor) -> torch.Tensor:
    """Stationary distribution of the hidden-state chain."""
    values, vectors = torch.linalg.eig(T.sum(0).T)
    pi = vectors[:, torch.argmin((values - 1).abs())].real
    return pi / pi.sum()


def sample(T: torch.Tensor, batch: int, length: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Token sequences (batch, length), each started from the stationary distribution."""
    pi = stationary_distribution(T)
    state = torch.multinomial(pi.expand(batch, 5), 1, generator=generator).squeeze(1)
    joint = T.permute(1, 0, 2).reshape(5, 10)  # joint[s, 5 x + t] = T[x, s, t]
    tokens = torch.empty(batch, length, dtype=torch.long)
    for i in range(length):
        outcome = torch.multinomial(joint[state], 1, generator=generator).squeeze(1)
        tokens[:, i], state = outcome // 5, outcome % 5
    return tokens


def beliefs(T: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Bayesian belief states (batch, length + 1, 5): the posterior over hidden states before the first token and
    after each prefix of the sequence, starting from the stationary distribution. Sequences must have positive
    probability (for eps = 0 an impossible token gives 0 / 0)."""
    belief = stationary_distribution(T).expand(len(tokens), 5)
    out = [belief]
    for i in range(tokens.shape[1]):
        unnormalized = torch.einsum("bs,bst->bt", belief, T[tokens[:, i]])
        belief = unnormalized / unnormalized.sum(1, keepdim=True)
        out.append(belief)
    return torch.stack(out, 1)


def next_token_probabilities(T: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
    """Optimal next-token distribution (..., 2) given belief states (..., 5)."""
    return belief @ T.sum(2).T


def optimal_predictions(T: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Bayes-optimal distribution of the next token after each prefix (batch, length, 2): entry n is the distribution
    of tokens[:, n + 1] given tokens[:, :n + 1], i.e. what a model should output at position n."""
    return next_token_probabilities(T, beliefs(T, tokens)[:, 1:])


def sequence_probabilities(T: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Probability of each sequence (batch,) under the stationary process, for sequences with positive probability."""
    belief = beliefs(T, tokens)[:, :-1]
    predicted = next_token_probabilities(T, belief).gather(-1, tokens[..., None]).squeeze(-1)
    return predicted.prod(-1)


def kl_to_model(optimal: torch.Tensor, logits: torch.Tensor) -> float:
    """Mean KL divergence from optimal next-token distributions (..., 2) to the model's (logits (..., 2))."""
    log_q = torch.log_softmax(logits.detach().cpu().double(), -1)
    return float((optimal * (optimal.clamp_min(1e-300).log() - log_q)).sum(-1).mean())


def test_set(T: torch.Tensor, length: int, size: int = 4096, seed: int = 12345):
    """A fixed held-out set of sequences and the optimal next-token distributions after each of their prefixes."""
    tokens = sample(T, size, length, torch.Generator().manual_seed(seed))
    return tokens, optimal_predictions(T, tokens)
