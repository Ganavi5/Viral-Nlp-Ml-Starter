# step2_kmer_tokenizer.py

def kmers(sequence, k=6):
    """
    Split a DNA sequence string into k-mers (substrings of length k).
    Example: "ATCGT" with k=3 -> ["ATC", "TCG", "CGT"]
    """
    sequence = sequence.upper().replace("N", "")  # clean (remove 'N') 
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Quick test
if __name__ == "__main__":
    seq = "ATGCGTAC"
    print("Original:", seq)
    print("6-mers:", kmers(seq, k=6))
