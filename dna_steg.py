"""
dna_steg.py
===========
DNA Steganography prototype using MT19937 (Python random module)
for position index generation.

This is the TARGET system — the one being attacked.
Used in: A Quantum Attack Pipeline on DNA Steganography

Encoding rule (Table 1):
  00 -> A
  01 -> C
  10 -> G
  11 -> T
"""

import random
import os

# ── Encoding tables ───────────────────────────────────────────────────────────

BITS_TO_BASE = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}
BASE_TO_BITS = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
BASES = ['A', 'C', 'G', 'T']

# ── Encoding ──────────────────────────────────────────────────────────────────

def text_to_binary(text):
    """Convert plaintext string to binary string."""
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_dna(binary):
    """Convert binary string to DNA sequence using coding rule."""
    if len(binary) % 2 != 0:
        binary += '0'
    return ''.join(BITS_TO_BASE[binary[i:i+2]] for i in range(0, len(binary), 2))

def dna_to_binary(dna):
    """Convert DNA sequence back to binary string."""
    return ''.join(BASE_TO_BITS[b] for b in dna)

def binary_to_text(binary):
    """Convert binary string back to plaintext."""
    chars = []
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return ''.join(chars)

# ── R-DNA generation ──────────────────────────────────────────────────────────

def generate_rdna(length, seed=None):
    """
    Generate random R-DNA strand of given length.
    Uses os.urandom for the cover strand — only position
    generation uses the vulnerable MT19937 PRNG.
    """
    rng = random.Random(seed) if seed else random.Random()
    return ''.join(rng.choice(BASES) for _ in range(length))

# ── Steganography — hide ──────────────────────────────────────────────────────

def hide_message(message, prng_seed, rdna_length_multiplier=10):
    """
    Hide a message in a DNA strand.

    Args:
        message: plaintext string to hide
        prng_seed: seed for MT19937 position generation (THE VULNERABILITY)
        rdna_length_multiplier: how many times longer the cover strand is

    Returns:
        stego_strand: the DNA strand containing the hidden message
        secret_positions: the insertion positions (the key)
        ddna: the encoded message DNA sequence
    """
    # Step 1: encode message to DNA
    binary = text_to_binary(message)
    ddna = binary_to_dna(binary)
    msg_len = len(ddna)

    # Step 2: generate R-DNA cover strand
    rdna_length = msg_len * rdna_length_multiplier
    rdna = generate_rdna(rdna_length)

    # Step 3: generate insertion positions using MT19937 (THE VULNERABLE STEP)
    prng = random.Random(prng_seed)
    total_length = rdna_length + msg_len
    all_positions = list(range(total_length))

    # Sample msg_len positions without replacement
    secret_positions = sorted(prng.sample(all_positions, msg_len))

    # Step 4: build stego strand
    stego = list(rdna)
    # Insert D'DNA bases at the secret positions
    # We build the strand by interleaving
    result = []
    rdna_idx = 0
    ddna_idx = 0
    pos_set = set(secret_positions)

    for i in range(total_length):
        if i in pos_set:
            result.append(ddna[ddna_idx])
            ddna_idx += 1
        else:
            if rdna_idx < len(rdna):
                result.append(rdna[rdna_idx])
                rdna_idx += 1

    stego_strand = ''.join(result)

    return stego_strand, secret_positions, ddna

# ── Steganography — extract (with key) ───────────────────────────────────────

def extract_message(stego_strand, secret_positions):
    """
    Extract hidden message from stego strand using the key (positions).

    Args:
        stego_strand: the DNA strand containing the hidden message
        secret_positions: the insertion positions (the key)

    Returns:
        recovered message string
    """
    ddna = ''.join(stego_strand[p] for p in sorted(secret_positions))
    binary = dna_to_binary(ddna)
    return binary_to_text(binary)

# ── Observable output (what attacker sees) ────────────────────────────────────

def get_observable_positions(prng_seed, n_observations, rdna_length_multiplier=10,
                              message_length=8):
    """
    Simulate an attacker observing multiple position outputs from the system.
    In practice, an attacker might observe multiple messages being hidden
    and record the position indices each time.

    Returns list of position samples (each is a sorted list of positions).
    """
    observations = []
    msg_len = message_length  # assume fixed message length for simplicity

    prng = random.Random(prng_seed)
    total_length = (msg_len * rdna_length_multiplier) + msg_len

    for _ in range(n_observations):
        positions = sorted(prng.sample(range(total_length), msg_len))
        observations.append(positions)

    return observations

# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DNA STEGANOGRAPHY PROTOTYPE")
    print("Using MT19937 (Python random) for position generation")
    print("=" * 60)

    MESSAGE = "HELLO"
    SEED = 42  # in practice this would be secret

    print(f"\nOriginal message : '{MESSAGE}'")
    print(f"PRNG seed        : {SEED}  (secret — attacker doesn't know this)")

    # Hide
    stego, positions, ddna = hide_message(MESSAGE, SEED)

    print(f"\nEncoded D'DNA    : {ddna}")
    print(f"Secret positions : {positions}")
    print(f"Stego strand     : {stego[:80]}..." if len(stego) > 80 else f"Stego strand     : {stego}")
    print(f"Strand length    : {len(stego)}")

    # Extract with key (legitimate recipient)
    recovered = extract_message(stego, positions)
    print(f"\nRecovered (with key) : '{recovered}'")
    print(f"Correct              : {recovered.strip(chr(0)) == MESSAGE}")

    print("\n" + "=" * 60)
    print("OBSERVABLE OUTPUT (what attacker can collect)")
    print("=" * 60)
    obs = get_observable_positions(SEED, n_observations=5)
    for i, o in enumerate(obs):
        print(f"Observation {i+1}: {o}")
    print("\nAttacker sees these positions but not the seed.")
    print("Attack goal: recover seed from position observations.")