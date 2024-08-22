import numpy as np

class SimplifiedEnigmaNetwork:
    def __init__(self, length, input_size, output_size, connection_density=0.1):
        self.length = length
        self.input_size = input_size
        self.output_size = output_size
        self.connection_density = connection_density
        self.default_threshold = 3.0

        assert self.input_size + self.output_size <= self.length, \
            (f"Length of EnigmaNetwork must exceed both input and output combined, "
             f"got {self.input_size} and {self.output_size}")

        self.thresholds = np.full(self.length, self.default_threshold, dtype=float)
        self.thresholds[:self.input_size] = 1.0

        self.potentials = np.zeros(self.length)
        self.connections = [[] for _ in range(self.length)]

        for i in range(self.length - self.output_size):
            n_connections = self.calculate_connections(i)
            available_range = range(1, self.length - i)
            if len(available_range) > 0:
                n_to_connect = min(n_connections, len(available_range))
                self.connections[i] = list(np.random.choice(
                    available_range,
                    size=n_to_connect,
                    replace=False
                ))
        print("Completed SimplifiedEnigmaNetwork initialization")

    def calculate_connections(self, node_index):
        mx = self.length
        diff = mx - node_index
        return max(int(diff * self.connection_density), 1)

    def forward(self, inputs):
        assert len(inputs) == self.input_size, f"Expected {self.input_size} inputs, got {len(inputs)}"

        self.potentials.fill(0)
        self.potentials[:self.input_size] = inputs
        operations = 0

        keep_going = True
        while keep_going:
            operations += 1
            keep_going = False
            active_nodes = self.potentials >= self.thresholds
            if np.any(active_nodes):
                keep_going = True
                for i in np.where(active_nodes)[0]:
                    for conn in self.connections[i]:
                        self.potentials[conn + i] += (1 * ((100 - operations**2)/100))
                self.potentials[active_nodes] = 0
        return self.potentials[-self.output_size:].copy()

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    network = SimplifiedEnigmaNetwork(length=100, input_size=4, output_size=1, connection_density=0.1)

    # Test the network with some sample inputs
    test_cases = [
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1]
    ]

    print("\nTesting the network:")
    for test_input in test_cases:
        output = network.forward(np.array(test_input))
        print(f"Input: {test_input}, Output: {output}, Rounded: {np.round(output).astype(int)}")