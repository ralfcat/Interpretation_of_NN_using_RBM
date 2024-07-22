from ucimlrepo import fetch_ucirepo

# Fetch dataset
gene_expression_cancer_rna_seq = fetch_ucirepo(id=401)

# Access data and metadata
X = gene_expression_cancer_rna_seq.data.features
y = gene_expression_cancer_rna_seq.data.targets
metadata = gene_expression_cancer_rna_seq.metadata
variables = gene_expression_cancer_rna_seq.variables

print(metadata)
print(variables)
