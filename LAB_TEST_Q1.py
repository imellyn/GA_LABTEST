# app.py
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

POP_SIZE       = 300            
CHROM_LEN      = 80              
N_GENERATIONS  = 50             
TARGET_ONES    = 40              
MAX_FITNESS    = 80              

TOURNAMENT_K   = 4
CROSSOVER_RATE = 0.85
MUTATION_RATE  = 1.0 / CHROM_LEN   # Expected ~1 mutation per chromosome

# Fitness Function 
def fitness(individual: np.ndarray) -> float:
    """Fitness is maximum (80) when number of 1s = 40, decreases as we move away"""
    ones = np.sum(individual)
    return MAX_FITNESS - abs(ones - TARGET_ONES)

#GA Core Operators 
def init_population() -> np.ndarray:
    return np.random.randint(0, 2, size=(POP_SIZE, CHROM_LEN), dtype=np.int8)

def tournament_selection(pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
    idxs = np.random.choice(len(pop), size=TOURNAMENT_K, replace=False)
    winner_idx = idxs[np.argmax(fits[idxs])]
    return pop[winner_idx].copy()

def single_point_crossover(p1: np.ndarray, p2: np.ndarray):
    if random.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    
    point = random.randint(1, CHROM_LEN-1)
    c1 = np.concatenate((p1[:point], p2[point:]))
    c2 = np.concatenate((p2[:point], p1[point:]))
    return c1, c2

def mutate(ind: np.ndarray) -> np.ndarray:
    for i in range(CHROM_LEN):
        if random.random() < MUTATION_RATE:
            ind[i] = 1 - ind[i]
    return ind

def run_genetic_algorithm():
    population = init_population()
    best_fitness_history = []
    best_individual = None
    best_fitness = -float('inf')

    for gen in range(N_GENERATIONS):
        fits = np.array([fitness(ind) for ind in population])
        
        # Track best
        best_idx = np.argmax(fits)
        current_best_f = fits[best_idx]
        best_fitness_history.append(float(current_best_f))
        
        if current_best_f > best_fitness:
            best_fitness = current_best_f
            best_individual = population[best_idx].copy()

        # Elitism: keep the best one
        next_pop = [best_individual.copy()]

        # Generate the rest
        while len(next_pop) < POP_SIZE:
            parent1 = tournament_selection(population, fits)
            parent2 = tournament_selection(population, fits)
            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_pop.extend([child1, child2])

        population = np.array(next_pop[:POP_SIZE], dtype=np.int8)

    return best_individual, best_fitness, best_fitness_history


st.set_page_config(page_title="Genetic Algorithm - 80-bit Target 40 Ones", layout="wide")

st.title("🧬 Genetic Algorithm Demo")
st.subheader("Find an 80-bit string with exactly 40 ones")

st.markdown("""
**Fixed Parameters:**
- Population size: **300**
- Chromosome length: **80** bits
- Generations: **50**
- Target: exactly **40** ones
- Maximum fitness: **80** (when ones = 40)
""")

if st.button("Run Evolution", type="primary"):
    with st.spinner("Evolving population..."):
        random.seed(42)           # for reproducibility
        np.random.seed(42)

        best_ind, best_fit, history = run_genetic_algorithm()

        ones = int(np.sum(best_ind))
        bitstring = "".join(map(str, best_ind))

        st.success(f"**Best fitness achieved: {best_fit:.1f}**")

        cols = st.columns([2, 1])
        with cols[0]:
            st.markdown("**Best solution found**")
            st.code(bitstring, language="text")
        with cols[1]:
            st.metric("Number of 1s", ones, delta=f"{ones - TARGET_ONES:+d}")
            st.metric("Number of 0s", 80 - ones)

        # Plot convergence
        st.subheader("Convergence Curve")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, N_GENERATIONS + 1), history, 'b-', linewidth=2.5)
        ax.axhline(y=MAX_FITNESS, color='r', linestyle='--', alpha=0.6, label='Theoretical max (80)')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        if ones == TARGET_ONES:
            st.success("🎯 Perfect solution found! Exactly 40 ones achieved.")
        elif best_fit >= 78:
            st.info("Very close to optimal solution ✓")
        else:
            st.warning("May need more generations or different random seed for better result.")