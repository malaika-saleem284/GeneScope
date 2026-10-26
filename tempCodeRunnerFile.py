
            if old_amino == new_amino:
                effect = "Silent (same amino acid)"
            elif new_amino == "*":
                effect = "Nonsense (protein stops early)"
            elif old_amino == "*":
                effect = "Stop lost (protein gets longer)"
            else:
                effect = "Missense (amino acid changed)"

        mutations.append(
            {
                "Position": index + 1,
                "Change": f"{old_base} → {new_base}",
                "Codon": f"{old_codon} → {new_codon}",
                "Amino acid": amino_acid_change,
                "Effect": effect,
            }
        )

    return mutations


def detect_gene(sequence):
    """Compare the sequence with every reference gene and return the best match."""
    aligner = make_aligner()
    best_gene = None
    best_similarity = 0

    for gene_name, reference in REFERENCE_GENES.items():
        alignment = aligner.align(sequence, reference)[0]
        similarity = alignment.counts().identities / max(len(sequence), len(reference)) * 100

        if similarity > best_similarity:
            best_gene = gene_name
            best_similarity = similarity

    return best_gene, best_similarity

def kmer_features(sequence):
    """Count every 3-letter word in the sequence and return the 64 shares."""
    counts = dict.fromkeys(ALL_KMERS, 0)
    total = max(len(sequence) - KMER_SIZE + 1, 1)

    for i in range(len(sequence) - KMER_SIZE + 1):
        counts[sequence[i:i + KMER_SIZE]] += 1

    return [counts[word] / total for word in ALL_KMERS]


@st.cache_resource
def train_gene_model():
    """Train the ML model once. It learns from slightly mutated copies of each gene."""
    features = []
    labels = []

    for gene_name, reference in REFERENCE_GENES.items():
        for _ in range(200):
            variant = list(reference)

            # Change up to 5% of the letters at random positions
            for _ in range(random.randint(0, len(reference) // 20)):
                variant[random.randrange(len(variant))] = random.choice("ATGC")

            features.append(kmer_features("".join(variant)))
            labels.append(gene_name)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model


def predict_gene(sequence):
    """Ask the ML model which gene this is. Returns the gene and its confidence (%)."""
    model = train_gene_model()
    probabilities = model.predict_proba([kmer_features(sequence)])[0]
    best = probabilities.argmax()
    return model.classes_[best], probabilities[best] * 100