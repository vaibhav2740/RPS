import time
from rps_playground.algorithms import ALL_ALGORITHM_CLASSES
from rps_playground.engine import Move

def profile_algorithms():
    print(f"Profiling {len(ALL_ALGORITHM_CLASSES)} algorithms...")
    results = []
    
    # Dummy history
    my_hist = [Move.ROCK] * 100
    opp_hist = [Move.PAPER] * 100
    
    for AlgoClass in ALL_ALGORITHM_CLASSES:
        algo = AlgoClass()
        name = algo.name
        
        start = time.perf_counter()
        try:
            # Run 100 calls to choose()
            for i in range(100):
                algo.choose(i + 100, my_hist, opp_hist)
            duration = (time.perf_counter() - start) * 1000 # ms
            avg_time = duration / 100
            results.append((name, avg_time))
        except Exception as e:
            results.append((name, -1)) # Error
            print(f"Error in {name}: {e}")

    # Sort by time descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- Slowest Algorithms (avg ms per move) ---")
    for name, avg in results[:20]:
        print(f"{name:<30}: {avg:.4f} ms")

if __name__ == "__main__":
    profile_algorithms()
