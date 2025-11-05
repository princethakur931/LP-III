class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

def Calculate_Probability(data):
    symbols = {}
    for element in data:
        if symbols.get(element) is None:
            symbols[element] = 1
        else:
            symbols[element] += 1
    return symbols

def Calculate_Codes(node, val=''):
    newVal = val + str(node.code)
    if node.left:
        Calculate_Codes(node.left, newVal)
    if node.right:
        Calculate_Codes(node.right, newVal)
    if not node.left and not node.right:
        codes[node.symbol] = newVal
    return codes

def Output_Encoded(data, coding):
    encoding_output = []
    for c in data:
        encoding_output.append(coding[c])
    return ''.join(encoding_output)

def Total_Gain(data, coding):
    before_compression = len(data) * 8
    after_compression = len(Output_Encoded(data, coding))
    print("Space usage before compression (in bits):", before_compression)
    print("Space usage after compression (in bits):", after_compression)

def Huffman_Encoding(data):
    print("Original input:", data)
    symbol_with_probs = Calculate_Probability(data)
    symbols = list(symbol_with_probs.keys())
    probabilities = list(symbol_with_probs.values())

    print("symbols:", symbols)
    print("probabilities:", probabilities)

    nodes = [Node(symbol_with_probs[symbol], symbol) for symbol in symbols]

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.prob)
        left = nodes[0]
        right = nodes[1]
        left.code = 0
        right.code = 1
        nodes.remove(left)
        nodes.remove(right)
        newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)
        nodes.append(newNode)

    global codes
    codes = {}
    huffman_encoding = Calculate_Codes(nodes[0])
    print("symbols with codes", huffman_encoding)
    Total_Gain(data, huffman_encoding)
    encoded_output = Output_Encoded(data, huffman_encoding)
    print("Encoded output", encoded_output)
    return encoded_output

data = "AABBBCCCCCDDEEEEE"
Huffman_Encoding(data)