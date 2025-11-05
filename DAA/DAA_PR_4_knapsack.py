def knapsack(values, weights, capacity):
    n = len(values)
    # Initialize DP table
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w],
                               values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find included items
    w = capacity
    included_items = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            included_items.append(i - 1)
            w -= weights[i - 1]

    included_items.reverse()
    included_values = [values[i] for i in included_items]
    included_weights = [weights[i] for i in included_items]

    return dp[n][capacity], included_items, included_values, included_weights

# Input number of items
n = int(input("Enter the number of items: "))

# Input values and weights
values = []
weights = []
print("Enter the value and weight of each item:")
for i in range(n):
    val = int(input(f" Value of item {i + 1}: "))
    wt = int(input(f" Weight of item {i + 1}: "))
    values.append(val)
    weights.append(wt)

# Input knapsack capacity
capacity = int(input("Enter the maximum capacity of the knapsack: "))

# Solve knapsack
max_value, included_indices, included_values, included_weights = knapsack(values, weights, capacity)

# Output results
print("\n✅ Maximum value in knapsack:", max_value)
print(" Items included in knapsack (bucket):")
for idx, val, wt in zip(included_indices, included_values, included_weights):
    print(f" - Item {idx + 1}: Value = {val}, Weight = {wt}")