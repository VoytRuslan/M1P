import torch

def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[n][m]

def cer(pred, target):
    if len(target) == 0:
        return 0.0

    dist = levenshtein(pred, target)
    return dist / len(target)

def wer(pred, target):
    pred_words = pred.split()
    target_words = target.split()

    if len(target_words) == 0:
        return 0.0

    dist = levenshtein(pred_words, target_words)
    return dist / len(target_words)

def greedy_decode(logits, encoder):
    probs = torch.softmax(logits, dim=-1)
    indices = torch.argmax(probs, dim=-1)  # (T, B)

    results = []

    for b in range(indices.shape[1]):
        seq = indices[:, b].cpu().numpy().tolist()
        text = encoder.decode(seq)
        print(text, end='->')
        results.append(text)
    print(results)
    return results
