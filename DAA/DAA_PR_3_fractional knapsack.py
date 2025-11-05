class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

def fractionalKnapsack(W, arr):
    # Sort items by value-to-weight ratio in descending order
    arr.sort(key=lambda x: x.value / x.weight, reverse=True)

    finalValue = 0.0  # Total value accumulated

    for item in arr:
        if item.weight <= W:
            # Take the whole item
            W -= item.weight
            finalValue += item.value
        else:
            # Take the fractional part of the item
            finalValue += item.value * W / item.weight
            break

    return finalValue

# 🧪 Example usage
if __name__ == "__main__":
    W = 50  # Capacity of knapsack
    arr = [Item(60, 10), Item(100, 20), Item(120, 30)]

    max_val = fractionalKnapsack(W, arr)
    print("Maximum value in knapsack =", max_val)