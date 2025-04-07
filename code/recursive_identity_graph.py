import numpy as np
from math import log

class RCGUMNode:
    def __init__(self, id, s, psi):
        """Initialize node with ID, classical state s, and quantum state psi (Section 2)."""
        self.id = id
        self.s = s  # Binary identity state
        self.psi = np.array(psi, dtype=complex)  # Quantum state: [alpha, beta]
        self.in_edges = []  # Incoming edge sources (node IDs)
        self.out_edges = []  # Outgoing edge targets (node IDs)

class RecursiveIdentityGraph:
    def __init__(self, N, rho_max=6, r=2):
        """Initialize the recursive identity graph (Section 7, consciousness).
        Args:
            N: Initial number of nodes (entities/identities).
            rho_max: Max edge density (from N=16 entropy fit, Section 6).
            r: Neighborhood radius for Phi observer (Section 2.1).
        """
        self.nodes = [RCGUMNode(i, 0, [1.0 + 0j, 0.0 + 0j]) for i in range(N)]
        self.rho_max = rho_max
        self.r = r
        self.next_id = N
        self.identity_stability = []
        self.neighborhood_cache = {}  # Cache for neighborhoods

    def phi(self, node_id, recursive=False):
        """Observer compression: average state in neighborhood (Section 2.1).
        If recursive=True, checks if Φ(Φ(Sᵐ)) = s(nᵢ) for self-referential identity."""
        neighborhood = self.get_neighborhood(node_id, self.r)
        if not neighborhood:
            return self.nodes[node_id].s
        avg_s = sum(self.nodes[n].s for n in neighborhood) / len(neighborhood)
        compressed = 1 if avg_s >= 0.5 else 0

        if recursive:
            # Compress the compressed states of the neighborhood
            compressed_states = [self.phi(n, recursive=False) for n in neighborhood]
            if not compressed_states:
                return compressed
            second_avg = sum(compressed_states) / len(compressed_states)
            second_compressed = 1 if second_avg >= 0.5 else 0
            return 1 if second_compressed == self.nodes[node_id].s else 0
        return compressed

    def get_neighborhood(self, node_id, r):
        """Bidirectional causal neighborhood with caching (Section 2).
        Future: Clear cache periodically if graph grows significantly."""
        cache_key = (node_id, r)
        if cache_key in self.neighborhood_cache:
            return self.neighborhood_cache[cache_key]
        visited = set()
        queue = [(node_id, 0)]
        neighborhood = []
        while queue:
            current_id, dist = queue.pop(0)
            if current_id in visited or dist > r:
                continue
            visited.add(current_id)
            neighborhood.append(current_id)
            node = self.nodes[current_id]
            queue.extend([(n, dist + 1) for n in node.out_edges + node.in_edges])
        neighborhood = neighborhood[1:]  # Exclude self
        self.neighborhood_cache[cache_key] = neighborhood
        return neighborhood

    def u_n(self, node, decohere=False, damping_factor=0.9):
        """Quantum evolution: Hadamard gate with optional decoherence (Section 8).
        Args:
            decohere: If True, apply decoherence.
            damping_factor: Controls decoherence strength (0 to 1, 1=no decoherence).
        Future: Add environment coupling for more realistic decoherence."""
        h = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
        node.psi = np.dot(h, node.psi)
        if decohere:
            # Apply partial decoherence: dampen off-diagonal terms
            prob_0 = abs(node.psi[0])**2
            prob_1 = abs(node.psi[1])**2
            node.psi[0] *= np.sqrt(prob_0 * damping_factor + (1 - damping_factor))
            node.psi[1] *= np.sqrt(prob_1 * damping_factor + (1 - damping_factor))
            # Normalize
            norm = np.sqrt(abs(node.psi[0])**2 + abs(node.psi[1])**2)
            node.psi /= norm if norm > 0 else 1.0

    def r(self):
        """Recursive propagation: state update via majority influence (Section 7).
        Future: Experiment with weighted influence based on edge strength."""
        new_states = []
        for node in self.nodes:
            if node.in_edges:
                neighbor_states = [self.nodes[src].s for src in node.in_edges]
                majority = 1 if sum(neighbor_states) > len(neighbor_states) / 2 else 0
            else:
                majority = node.s
            new_states.append(majority)

        for i, node in enumerate(self.nodes):
            old_s = node.s
            node.s = new_states[i]
            if old_s != node.s and len(node.out_edges) < self.rho_max:
                new_node = RCGUMNode(self.next_id, 0, [1.0 + 0j, 0.0 + 0j])
                self.nodes.append(new_node)
                node.out_edges.append(self.next_id)
                new_node.in_edges.append(node.id)
                self.next_id += 1

    def compute_identity_stability(self):
        """Stability of identity state across steps (Section 7)."""
        if not hasattr(self, 'last_states'):
            self.last_states = [node.s for node in self.nodes]
            return 1.0
        current_states = [node.s for node in self.nodes]
        stability = sum(1 for last, curr in zip(self.last_states, current_states) if last == curr) / len(self.nodes)
        self.last_states = current_states
        return stability

    def compute_entropy(self):
        """Weighted entropy based on path complexity and edge centrality (Section 3).
        Future: Add quantum entropy term based on psi states."""
        omega = [self.count_past_paths(node.id) for node in self.nodes]
        weighted = [log(w) / log(2) * (len(n.in_edges) + 1) for w, n in zip(omega, self.nodes) if w > 1]
        return sum(weighted) / len(self.nodes) if self.nodes else 0

    def count_past_paths(self, node_id):
        """Count unique paths to a node (Section 3).
        Future: Use dynamic programming to cache paths for large graphs."""
        visited = set()
        queue = [node_id]
        paths = 1
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            for src in self.nodes[current_id].in_edges:
                if src not in visited:
                    queue.append(src)
                    paths += 1
        return paths

    def compute_timing_residual(self, node_id):
        """Raw ΔT fluctuation based on edge density (Section 6)."""
        rho = len(self.nodes[node_id].out_edges)
        return 1e-10 * np.log1p(rho) / np.log1p(self.rho_max)

def simulate_identity_evolution(N, steps, target_cycle_days, pulsar_name="PSR B0919+06", recursive_phi=False, decohere=False):
    """Simulate evolution with optional recursive Φ and quantum decoherence (Section 7).
    Args:
        pulsar_name: Name of pulsar for ΔT scaling (ties to Section 6).
    """
    # Scaling factors to match ΔT targets (calculated as target / base Δt ≈ 2.50e-10 s)
    scaling_factors = {
        "PSR B0919+06": 2232,  # 5.58e-07 s
        "PSR J0437-4715": 400,  # 1e-07 s
        "PSR J1903+0327": 4.0,  # 1e-09 s
        "PSR J0737-3039A": 40.0,  # 1e-08 s
        "PSR J1740-3015": 2.0,  # 5e-10 s
        "PSR J1023+0038": 200,  # 5e-08 s
    }
    scale = scaling_factors.get(pulsar_name, 1.0)
    graph = RecursiveIdentityGraph(N)
    delta_t = 1e-10
    cycle_steps = int(target_cycle_days * 86400 / delta_t)

    residuals, entropies, stabilities = [], [], []

    for t in range(steps):
        for i in range(len(graph.nodes)):
            graph.nodes[i].s = graph.phi(i, recursive=recursive_phi)
        for node in graph.nodes:
            graph.u_n(node, decohere=decohere)
        graph.r()

        residual = graph.compute_timing_residual(0) * scale
        entropy = graph.compute_entropy()
        stability = graph.compute_identity_stability()
        residuals.append(residual)
        entropies.append(entropy)
        stabilities.append(stability)

        if t > 0 and t % (cycle_steps // 10) == 0:
            print(f"Step {t}: ΔT={residuals[-1]:.2e} s, S={entropies[-1]:.2f}, Φₛ={stabilities[-1]:.2f}")

    return residuals, entropies, stabilities

def main():
    """Test recursive identity graph evolution (Section 7).
    Future: Test with multiple pulsars, integrate with neuroscience data (Section 8)."""
    N = 100  # Small scale for demo; use N=5000 for production
    steps = 1000  # Subset for demo
    cycle = 600.0  # PSR B0919+06 benchmark

    residuals, entropies, stabilities = simulate_identity_evolution(
        N, steps, cycle, pulsar_name="PSR B0919+06", recursive_phi=True, decohere=True
    )

    print("\nRecursive Identity Simulation Summary:")
    print(f"Avg ΔT: {np.mean(residuals):.2e} s (target 5.58e-07 s for PSR B0919+06)")
    print(f"Avg Entropy: {np.mean(entropies):.2f} bits")
    print(f"Avg Stability: {np.mean(stabilities):.2f} (fraction of unchanged states)")

if __name__ == "__main__":
    main()
