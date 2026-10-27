import random
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from Bio import Align, Entrez, Phylo, SeqIO
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from sklearn.ensemble import RandomForestClassifier

# Only these letters are valid in a DNA sequence
VALID_LETTERS = set("ATGC")

# Smallest ORF to report (in amino acids)
MIN_ORF_PROTEIN_LENGTH = 3

# NCBI asks for an email so they can contact you if there is a problem.
# Replace this with your own email.
NCBI_EMAIL = "xyz.com"

# Normal (healthy) gene sequences. Positions count from the A of ATG.
REFERENCE_GENES = {
    "HBB": (
        "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAAC"
        "GTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAG"
        "AGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAG"
        "GTGAAGGCCCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGAC"
        "AACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGAT"
        "CCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGCGTGCTGGCCCATCACTTTGGC"
        "AAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAAT"
        "GCCCTGGCCCACAAGTATCACTAA"
    ),
    "HBA1": (
        "ATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTCGGCGCG"
        "CACGCTGGCGAGTATGGTGCGGAGGCCCTGGAGAGGATGTTCCTGTCCTTCCCCACCACC"
        "AAGACCTACTTCCCGCACTTCGACCTGAGCCACGGCTCTGCCCAGGTTAAGGGCCACGGC"
        "AAGAAGGTGGCCGACGCGCTGACCAACGCCGTGGCGCACGTGGACGACATGCCCAACGCG"
        "CTGTCCGCCCTGAGCGACCTGCACGCGCACAAGCTTCGGGTGGACCCGGTCAACTTCAAG"
        "CTCCTAAGCCACTGCCTGCTGGTGACCCTGGCCGCCCACCTCCCCGCCGAGTTCACCCCT"
        "GCGGTGCACGCCTCCCTGGACAAGTTCCTGGCTTCTGTGAGCACCGTGCTGACCTCCAAA"
        "TACCGTTAA"
    ),
    "INS": (
        "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGAC"
        "CCAGCCGCAGCCTTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTAC"
        "CTAGTGTGCGGGGAACGAGGCTTCTTCTACACACCCAAGACCCGCCGGGAGGCAGAGGAC"
        "CTGCAGGTGGGGCAGGTGGAGCTGGGCGGGGGCCCTGGTGCAGGCAGCCTGCAGCCCTTG"
        "GCCCTGGAGGGGTCCCTGCAGAAGCGTGGCATTGTGGAACAATGCTGTACCAGCATCTGC"
        "TCCCTCTACCAGCTGGAGAACTACTGCAACTAG"
    ),
}

# A sequence must be at least this similar (in %) to a reference gene to be accepted
MIN_GENE_SIMILARITY = 80

# The ML model counts words of this many letters (3 letters = 64 possible words)
KMER_SIZE = 3
ALL_KMERS = ["".join(word) for word in product("ATGC", repeat=KMER_SIZE)]

# Known disease mutations: (gene, position, change) -> disease
KNOWN_MUTATIONS = {
    ("HBB", 20, "A → T"): "Sickle Cell Anemia",
    ("INS", 287, "G → A"): "Permanent Neonatal Diabetes",
}


def explain(title, text):
    """Show a small collapsible box that explains a result."""
    with st.expander(f"ℹ️ {title}"):
        st.write(text)


def to_csv(table):
    """Convert a table to CSV bytes for the download button."""
    return table.to_csv(index=False).encode("utf-8")


def fetch_sequence(accession):
    """Download one sequence record from the NCBI nucleotide database."""
    Entrez.email = NCBI_EMAIL
    with Entrez.efetch(
        db="nucleotide", id=accession, rettype="fasta", retmode="text"
    ) as handle:
        return SeqIO.read(handle, "fasta")


def clean_sequence(raw_text):
    """Remove spaces, make uppercase, and convert RNA (U) to DNA (T)."""
    return "".join(raw_text.split()).upper().replace("U", "T")


def make_aligner():
    """Create an aligner. Scoring: same letter +1, different letter -1, gap -2."""
    aligner = Align.PairwiseAligner()
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -1
    return aligner


def parse_fasta(raw_text):
    """Read FASTA text (>name line, then sequence) and return names and sequences."""
    names = []
    sequences = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            # A new sequence starts here
            names.append(line[1:].strip())
            sequences.append("")
        elif sequences:
            # Add this line to the current sequence
            sequences[-1] += line

    return names, [clean_sequence(sequence) for sequence in sequences]


def build_tree(names, sequences):
    """Build a phylogenetic tree (UPGMA) from the distances between sequences."""
    aligner = make_aligner()
    matrix = []

    # Lower-triangle distance matrix: distance = 1 - similarity
    for i in range(len(sequences)):
        row = []
        for j in range(i):
            alignment = aligner.align(sequences[i], sequences[j])[0]
            longest = max(len(sequences[i]), len(sequences[j]))
            row.append(1 - alignment.counts().identities / longest)
        row.append(0)
        matrix.append(row)

    distance_matrix = DistanceMatrix(names, matrix)
    return DistanceTreeConstructor().upgma(distance_matrix)


def find_orfs(sequence, min_length=MIN_ORF_PROTEIN_LENGTH):
    """Find ORFs (ATG ... stop) in the 3 forward reading frames."""
    orfs = []

    for frame in range(3):
        # Keep only complete groups of 3 letters, starting from this frame
        usable_length = (len(sequence) - frame) // 3 * 3
        protein = str(Seq(sequence[frame:frame + usable_length]).translate())

        # Position (in amino acids) where the current part begins
        part_start = 0

        # Every part except the last one ends with a stop signal (*)
        for part in protein.split("*")[:-1]:
            first_m = part.find("M")

            if first_m != -1:
                orf_protein = part[first_m:]

                if len(orf_protein) >= min_length:
                    # 1-based positions; the stop codon is included in the end
                    start = frame + (part_start + first_m) * 3 + 1
                    end = start + (len(orf_protein) + 1) * 3 - 1
                    orfs.append(
                        {
                            "Frame": frame + 1,
                            "Start": start,
                            "End": end,
                            "Length (nt)": end - start + 1,
                            "Protein": orf_protein,
                        }
                    )

            # Move past this part and its stop signal
            part_start += len(part) + 1

    return orfs


def find_mutations(sequence_1, sequence_2):
    """Compare two equal-length sequences letter by letter and describe each change."""
    mutations = []

    for index, (old_base, new_base) in enumerate(zip(sequence_1, sequence_2)):
        # Skip positions where both sequences have the same letter
        if old_base == new_base:
            continue

        # The 3-letter codon that contains this position
        codon_start = index // 3 * 3
        old_codon = sequence_1[codon_start:codon_start + 3]
        new_codon = sequence_2[codon_start:codon_start + 3]

        if len(old_codon) < 3:
            # Last letters that do not form a full codon
            amino_acid_change = "-"
            effect = "Not in a full codon"
        else:
            old_amino = str(Seq(old_codon).translate())
            new_amino = str(Seq(new_codon).translate())
            amino_acid_change = f"{old_amino} → {new_amino}"

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

def match_known_mutations(gene, mutations):
    """Find which mutations are known to cause a disease."""
    matches = []

    for mutation in mutations:
        key = (gene, mutation["Position"], mutation["Change"])

        if key in KNOWN_MUTATIONS:
            matches.append(
                {
                    "Gene": gene,
                    "Position": mutation["Position"],
                    "Change": mutation["Change"],
                    "Disease": KNOWN_MUTATIONS[key],
                }
            )

    return matches


# Page settings (browser tab title, icon, layout)
st.set_page_config(page_title="GeneScope", page_icon="🧬", layout="centered")

# Page heading and description
st.title("🧬 GeneScope")
st.write("A web-based tool for DNA and RNA sequence analysis")

# ---------------------------------------------------------------
# Fetch a sequence from NCBI (this must come before the input box)
# ---------------------------------------------------------------
accession = st.text_input(
    "Fetch a sequence from NCBI (accession ID):",
    placeholder="e.g. NC_045512.2 (COVID-19 reference genome)",
)

if st.button("Fetch from NCBI"):
    if not accession.strip():
        st.warning("Enter an accession ID first.")
    else:
        try:
            with st.spinner("Fetching from NCBI..."):
                record = fetch_sequence(accession.strip())

            # Put the sequence into the input box below
            st.session_state["main_sequence"] = str(record.seq)
            st.success(f"Loaded: {record.description} ({len(record.seq)} letters). Now click Analyze.")
        except Exception as error:
            st.error(f"Could not fetch this sequence. Check the ID and your internet. ({error})")

# Input box where the user pastes the sequence
raw_sequence = st.text_area(
    "Enter your DNA or RNA sequence (example: ATGCGTACGTTAGC):",
    height=150,
    key="main_sequence",
)

# Everything below runs only when the Analyze button is clicked
if st.button("Analyze", type="primary"):
    # Remove spaces/new lines and convert to uppercase
    sequence = "".join(raw_sequence.split()).upper()

    # Check if the input is RNA (contains U)
    is_rna = "U" in sequence

    if not sequence:
        # Empty input
        st.warning("Enter a sequence first.")
    elif is_rna and "T" in sequence:
        # A real sequence is either DNA (T) or RNA (U), never both
        st.error("Invalid sequence. Use either T (DNA) or U (RNA), not both.")
    else:
        # Convert RNA to DNA form so the rest of the code works for both
        sequence = sequence.replace("U", "T")

        if not set(sequence).issubset(VALID_LETTERS):
            # Input contains letters other than A, T, G, C (or U)
            st.error("Invalid sequence. Use only the letters A, T, G, C (or U for RNA).")
        else:
            if is_rna:
                st.info("RNA sequence detected. U was converted to T for analysis.")

            # GC content as a percentage
            gc_content = gc_fraction(sequence) * 100

            # Count how many times each nucleotide appears
            counts = {
                "A": sequence.count("A"),
                "T": sequence.count("T"),
                "G": sequence.count("G"),
                "C": sequence.count("C"),
            }

            # Show length and GC content side by side
            col1, col2 = st.columns(2)
            col1.metric("Sequence length", len(sequence))
            col2.metric("GC content", f"{gc_content:.2f}%")
            explain(
                "What is GC content?",
                "GC content is the percentage of G and C bases in the sequence. "
                "G-C pairs have 3 hydrogen bonds, while A-T pairs have only 2, "
                "so DNA with a higher GC content is more stable.",
            )

            # Bar chart of nucleotide counts
            st.subheader("Nucleotide count")
            st.bar_chart(pd.Series(counts))
            explain(
                "What is nucleotide count?",
                "It shows how many times each base (A, T, G and C) appears in the sequence.",
            )

            # Convert the text into a Biopython Seq object
            dna = Seq(sequence)

            # Reverse complement (the opposite DNA strand)
            st.subheader("Reverse complement")
            st.code(str(dna.reverse_complement()))
            explain(
                "What is a reverse complement?",
                "It is the sequence of the opposite DNA strand. Each base is replaced "
                "by its partner (A with T, G with C), and then the result is reversed.",
            )

            # Transcription (DNA to RNA, T becomes U)
            st.subheader("RNA (transcription)")
            st.code(str(dna.transcribe()))
            explain(
                "What is transcription?",
                "Transcription makes RNA from DNA. The only change is that every T becomes U.",
            )

            # Protein translation (every 3 letters make one amino acid)
            st.subheader("Protein translation")
            leftover = len(sequence) % 3
            usable_length = len(sequence) - leftover

            if usable_length == 0:
                st.info("Sequence is too short to translate. Enter at least 3 letters.")
            else:
                st.code(str(dna[:usable_length].translate()))
                st.write("A star (*) means a stop signal.")
                if leftover:
                    st.write(f"Last {leftover} letter(s) were ignored (a codon needs 3 letters).")
            explain(
                "What is translation?",
                "Every 3 letters (a codon) make one amino acid, and amino acids join to "
                "form a protein. A star (*) is a stop signal where the protein ends.",
            )

            # ORF finder (possible genes), longest ORFs first
            st.subheader("ORF finder")
            orfs = find_orfs(sequence)

            if orfs:
                orf_table = pd.DataFrame(orfs).sort_values("Length (nt)", ascending=False)
                st.dataframe(orf_table, hide_index=True)
                st.download_button(
                    "Download ORFs (CSV)",
                    data=to_csv(orf_table),
                    file_name="orfs.csv",
                    mime="text/csv",
                    on_click="ignore",
                )
            else:
                st.info("No ORF found (needs ATG followed by a stop codon).")
            explain(
                "What is an ORF?",
                "An Open Reading Frame (ORF) starts with ATG and ends with a stop codon "
                "(TAA, TAG or TGA). It marks a region that may contain a gene. "
                "This tool searches the 3 forward reading frames.",
            )

# ---------------------------------------------------------------
# Compare two sequences
# ---------------------------------------------------------------
st.divider()
st.header("Compare two sequences")
st.write("Find out how similar two sequences are and what changed.")

raw_sequence_1 = st.text_area("Sequence 1 (original):", height=100)
raw_sequence_2 = st.text_area("Sequence 2 (changed):", height=100)

if st.button("Compare", type="primary"):
    sequence_1 = clean_sequence(raw_sequence_1)
    sequence_2 = clean_sequence(raw_sequence_2)

    if not sequence_1 or not sequence_2:
        st.warning("Enter both sequences first.")
    elif not set(sequence_1 + sequence_2).issubset(VALID_LETTERS):
        st.error("Invalid sequence. Use only the letters A, T, G, C (or U for RNA).")
    else:
        # Take the best alignment
        best = make_aligner().align(sequence_1, sequence_2)[0]

        # Similarity = same letters out of the longer sequence
        identities = best.counts().identities
        similarity = identities / max(len(sequence_1), len(sequence_2)) * 100

        col1, col2 = st.columns(2)
        col1.metric("Similarity", f"{similarity:.2f}%")
        col2.metric("Alignment score", f"{best.score:.0f}")
        explain(
            "What are similarity and alignment score?",
            "Similarity is the percentage of positions that have the same letter. "
            "The alignment score gives +1 for a match, -1 for a mismatch and -2 for a gap. "
            "A higher score means the sequences are more alike.",
        )

        st.subheader("Alignment")
        st.code(str(best))
        explain(
            "How to read the alignment?",
            "The two sequences are placed one above the other. A | means the letters "
            "are the same, and a . means they are different.",
        )

        # Mutation finder (needs two sequences of the same length)
        st.subheader("Mutations")
        if len(sequence_1) != len(sequence_2):
            st.info("Mutation finder needs two sequences of the same length.")
        else:
            mutations = find_mutations(sequence_1, sequence_2)

            if mutations:
                mutation_table = pd.DataFrame(mutations)
                st.dataframe(mutation_table, hide_index=True)
                st.write("Codons are counted from the first letter of Sequence 1.")
                st.download_button(
                    "Download mutations (CSV)",
                    data=to_csv(mutation_table),
                    file_name="mutations.csv",
                    mime="text/csv",
                    on_click="ignore",
                )
            else:
                st.success("No mutations found. The sequences are identical.")
        explain(
            "What is a mutation?",
            "A mutation is a change in the sequence. Silent: the amino acid stays the same. "
            "Missense: the amino acid changes. Nonsense: a stop codon appears, "
            "so the protein ends early.",
        )

# ---------------------------------------------------------------
# Phylogenetic tree
# ---------------------------------------------------------------
st.divider()
st.header("Phylogenetic tree")
st.write("Paste 3 or more sequences in FASTA format. Each one starts with a > name line.")
st.code(">Original\nATGGCCATTGTAATGGGCCGC\n>Variant_A\nATGGCCATTGTAATGGGCCGA")

raw_fasta = st.text_area("Sequences (FASTA format):", height=200)

if st.button("Build tree", type="primary"):
    names, tree_sequences = parse_fasta(raw_fasta)

    if len(names) < 3:
        st.warning("Enter at least 3 sequences, each starting with a > name line.")
    elif len(set(names)) != len(names):
        st.error("Each sequence needs a different name.")
    elif any(not tree_sequence for tree_sequence in tree_sequences):
        st.error("Every name needs a sequence under it.")
    elif not set("".join(tree_sequences)).issubset(VALID_LETTERS):
        st.error("Invalid sequence. Use only the letters A, T, G, C (or U for RNA).")
    else:
        tree = build_tree(names, tree_sequences)

        # Draw the tree; show names only on the tips
        figure, axes = plt.subplots()
        Phylo.draw(
            tree,
            axes=axes,
            do_show=False,
            label_func=lambda clade: clade.name if clade.is_terminal() else "",
        )
        st.pyplot(figure)
        explain(
            "How to read the tree?",
            "Sequences that are more similar sit closer together on the tree. "
            "Long branches mean bigger differences. This is how scientists "
            "show how virus variants are related.",
        )


# ---------------------------------------------------------------
# Disease detector
# ---------------------------------------------------------------
st.divider()
st.header("Disease detector")
st.write("Paste a gene sequence. The app finds which gene it is and checks it for known disease mutations.")

raw_gene = st.text_area("Gene sequence:", height=150, key="gene_input")

if st.button("Detect disease", type="primary"):
    gene_sequence = clean_sequence(raw_gene)

    if not gene_sequence:
        st.warning("Enter a sequence first.")

    elif not set(gene_sequence).issubset(VALID_LETTERS):
        st.error("Invalid sequence. Use only the letters A, T, G, C (or U for RNA).")

    else:
        gene_name, gene_similarity = detect_gene(gene_sequence)

        if gene_similarity < MIN_GENE_SIMILARITY:
            st.warning("This gene is not in the reference list, so it cannot be checked.")

        else:
            reference = REFERENCE_GENES[gene_name]
            st.info(f"Detected gene: {gene_name} ({gene_similarity:.1f}% match)")

            # Second opinion from the ML model
            ml_gene, ml_confidence = predict_gene(gene_sequence)

            if ml_gene == gene_name:
                st.success(
                    f"ML model agrees: {ml_gene} ({ml_confidence:.0f}% confidence)"
                )
            else:
                st.warning(
                    f"ML model disagrees: it predicts {ml_gene} "
                    f"({ml_confidence:.0f}% confidence)"
                )

            explain(
                "What is the ML model?",
                "A Random Forest model learned the patterns of 3-letter words (like ATG or GCC) "
                "in each gene. It gives a second opinion, so the gene is confirmed in two ways: "
                "similarity and machine learning.",
            )

            if len(gene_sequence) != len(reference):
                st.info(
                    f"The mutation check needs the full {gene_name} "
                    f"coding sequence ({len(reference)} letters)."
                )

            else:
                mutations = find_mutations(reference, gene_sequence)
                matches = match_known_mutations(gene_name, mutations)

                # Share of the gene that is changed (in percent)
                mutated_percent = len(mutations) / len(reference) * 100

                if matches:
                    st.error("Known disease mutation found.")
                    st.dataframe(pd.DataFrame(matches), hide_index=True)

                elif mutations:
                    st.warning("Mutations found, but none match a known disease.")
                    st.dataframe(pd.DataFrame(mutations), hide_index=True)

                else:
                    st.success(
                        f"No mutations found. This matches the normal {gene_name} gene."
                    )

                # Graph: how much of the gene is normal and how much is mutated
                st.subheader("Mutated percentage")

                col1, col2 = st.columns(2)
                col1.metric("Mutations found", len(mutations))
                col2.metric("Mutated", f"{mutated_percent:.2f}%")

                chart_data = pd.DataFrame(
                    {
                        "Normal": [100 - mutated_percent],
                        "Mutated": [mutated_percent],
                    },
                    index=["Gene"],
                )

                st.bar_chart(chart_data, color=["#4CAF50", "#FF6B6B"])

                explain(
                    "How to read this graph?",
                    "It shows how much of the gene is the same as the normal gene and how much "
                    "is changed. The mutated part is often very small, but even one change in an "
                    "important place can cause a disease.",
                )