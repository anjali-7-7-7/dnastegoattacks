"""
attacker.py
===========
MT19937 seed recovery attack against DNA steganography.
Demonstrates that PRNG-based position generation is exploitable.
"""

import random
from dna_steg import hide_message, extract_message

def recover_seed_brute(observed_positions, total_length, msg_len, seed_range=100000):
    """
    Recover MT19937 seed by brute force over seed_range.
    For demonstration — in practice ML preprocessing narrows this space.
    """
    print(f"Searching seeds 0 to {seed_range}...")
    for candidate_seed in range(seed_range):
        prng = random.Random(candidate_seed)
        candidate = sorted(prng.sample(range(total_length), msg_len))
        if candidate == observed_positions:
            return candidate_seed
    return None

def attack(message, true_seed, mult=10):
    print("=" * 60)
    print("ATTACK — MT19937 SEED RECOVERY")
    print("=" * 60)

    # Step 1: target hides a message
    stego, true_positions, ddna = hide_message(message, true_seed, mult)
    msg_len = len(ddna)
    total_length = len(stego)

    print(f"\nTarget message length : {len(message)} chars")
    print(f"D'DNA length          : {msg_len} bases")
    print(f"Stego strand length   : {total_length} bases")
    print(f"True seed (hidden)    : {true_seed}")
    print(f"True positions        : {true_positions}")

    # Step 2: attacker observes positions from one interception
    observed = true_positions
    print(f"\nAttacker observes     : {observed}")

    # Step 3: brute force seed recovery
    print(f"\nRunning seed recovery...")
    import time
    t0 = time.time()
    recovered_seed = recover_seed_brute(observed, total_length, msg_len)
    elapsed = time.time() - t0

    if recovered_seed is not None:
        print(f"\n✓ SEED RECOVERED: {recovered_seed}")
        print(f"  Time taken: {elapsed:.3f}s")
        print(f"  Seed correct: {recovered_seed == true_seed}")

        # Step 4: reconstruct positions and extract message
        prng = random.Random(recovered_seed)
        reconstructed_positions = sorted(prng.sample(range(total_length), msg_len))
        recovered_message = extract_message(stego, reconstructed_positions)

        print(f"\n✓ POSITIONS RECONSTRUCTED: {reconstructed_positions}")
        print(f"✓ MESSAGE EXTRACTED: '{recovered_message.strip(chr(0))}'")
        print(f"✓ CORRECT: {recovered_message.strip(chr(0)) == message}")
        print(f"\nThe DNA strand was never analysed.")
        print(f"The attack targeted the PRNG seed — not the DNA.")
    else:
        print("Seed not found in range — increase seed_range")

    return recovered_seed

if __name__ == "__main__":
    # Demo attack with known seed in small range
    attack("HELLO", true_seed=42)

    print("\n" + "=" * 60)
    print("WHAT THIS PROVES")
    print("=" * 60)
    print("1. PRNG seed is the real attack surface — not the DNA")
    print("2. Seed recovery reconstructs all future/past positions")
    print("3. Message extracted without touching the biochemical layer")
    print("4. Classical brute force works against weak seeds")
    print("5. Quantum Grover's provides quadratic speedup for larger seeds")
    print("   e.g. 2^32 seed space -> 2^16 Grover operations")
