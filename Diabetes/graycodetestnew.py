import sys
from collections import defaultdict

def min_draws_to_reach_end(N, colors):
    # Initialize DP array with a large value (infinity)
    DP = [float('inf')] * N
    pos = {}

    # Start from the last position
    pos[colors[-1]] = N - 1
    DP[-1] = 0

    # Iterate from the second last square to the first
    for i in range(N - 2, -1, -1):
        d = DP[i]
        for color, j in pos.items():
            d = min(d, DP[j] + 1)
        DP[i] = d
        pos[colors[i]] = i

    # Find the minimum number of moves to reach the last square
    res = float('inf')
    for color, j in pos.items():
        if DP[j] != float('inf'):
            res = min(res, DP[j] + 1)

    return res

# Test cases
if __name__ == "__main__":
    test_cases = [
        (6, ["Blue", "Orange", "Pink", "Green", "Red", "Yellow"]),
        (12, ["Blue", "Orange", "Pink", "Green", "Red", "Yellow", "Yellow", "Red", "Green", "Pink", "Orange", "Blue"]),
        (9, ["Blue", "Orange", "Pink", "Green", "Red", "Yellow", "Yellow", "Yellow", "Yellow"]),
    ]

    expected_outputs = [1, 2, 4]

    for i, (N, colors) in enumerate(test_cases):
        result = min_draws_to_reach_end(N, colors)
        print(f"Test Case {i+1}: Expected = {expected_outputs[i]}, Got = {result}")
        assert result == expected_outputs[i], f"Test case {i+1} failed"
    print("All test cases passed!")
