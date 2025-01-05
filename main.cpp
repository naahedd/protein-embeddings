#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <string>

std::map<char, int> amino_acid_to_id = {
    {'A', 0}, {'C', 1}, {'D', 2}, {'E', 3}, {'F', 4}, {'G', 5}, {'H', 6}, {'I', 7}, {'K', 8}, {'L', 9},
    {'M', 10}, {'N', 11}, {'P', 12}, {'Q', 13}, {'R', 14}, {'S', 15}, {'T', 16}, {'V', 17}, {'W', 18}, {'Y', 19},
    {'-', 20}
};

std::string amino_acids = "ACDEFGHIKLMNPQRSTVWY";

std::string introduce_mutations(const std::string& sequence, double mutation_rate = 0.1, double insertion_rate = 0.05, double deletion_rate = 0.05) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::string modified_sequence;
    for (char amino_acid : sequence) {
        if (dis(gen) < deletion_rate) continue;
        if (dis(gen) < mutation_rate) amino_acid = amino_acids[std::uniform_int_distribution<>(0, amino_acids.size() - 1)(gen)];
        modified_sequence += amino_acid;
        if (dis(gen) < insertion_rate) modified_sequence += amino_acids[std::uniform_int_distribution<>(0, amino_acids.size() - 1)(gen)];
    }
    return modified_sequence;
}

std::vector<std::string> simulate_msa(const std::string& input_sequence, int num_sequences = 10, double mutation_rate = 0.1, double insertion_rate = 0.05, double deletion_rate = 0.05) {
    std::vector<std::string> msa = {input_sequence};
    for (int i = 1; i < num_sequences; ++i) {
        msa.push_back(introduce_mutations(input_sequence, mutation_rate, insertion_rate, deletion_rate));
    }
    return msa;
}

torch::Tensor convert_to_tensor(const std::vector<std::string>& msa, const std::map<char, int>& amino_acid_to_id) {
    int num_sequences = msa.size();
    int sequence_length = 0;
    for (const auto& seq : msa) {
        if (seq.size() > sequence_length) sequence_length = seq.size();
    }

    auto options = torch::TensorOptions().dtype(torch::kLong);
    torch::Tensor msa_tensor = torch::full({num_sequences, sequence_length}, amino_acid_to_id.at('-'), options);

    for (int i = 0; i < num_sequences; ++i) {
        for (int j = 0; j < msa[i].size(); ++j) {
            char amino_acid = msa[i][j];
            msa_tensor[i][j] = amino_acid_to_id.count(amino_acid) ? amino_acid_to_id.at(amino_acid) : amino_acid_to_id.at('-');
        }
    }
    return msa_tensor;
}

torch::Tensor prepare_msa_tensor(const std::string& input_sequence) {
    auto msa = simulate_msa(input_sequence);
    return convert_to_tensor(msa, amino_acid_to_id);
}

struct MSAEmbeddingModel : torch::nn::Module {
    torch::nn::Embedding embedding;
    torch::nn::TransformerEncoder transformer;

    MSAEmbeddingModel(int num_amino_acids, int embedding_dim, int nhead, int num_layers)
        : embedding(torch::nn::EmbeddingOptions(num_amino_acids, embedding_dim)),
          transformer(torch::nn::TransformerEncoderOptions(torch::nn::TransformerEncoderLayerOptions(embedding_dim, nhead), num_layers)) {
        register_module("embedding", embedding);
        register_module("transformer", transformer);
    }

    torch::Tensor forward(torch::Tensor msa_tensor) {
        auto embedded = embedding(msa_tensor);
        embedded = embedded.permute({1, 0, 2});
        auto msa_embeddings = transformer(embedded);
        msa_embeddings = msa_embeddings.permute({1, 0, 2});
        return msa_embeddings;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <sequence>" << std::endl;
        return 1;
    }

    std::string sequence = argv[1];
    int num_amino_acids = 21;
    int embedding_dim = 64;
    int nhead = 8;
    int num_layers = 4;

    MSAEmbeddingModel model(num_amino_acids, embedding_dim, nhead, num_layers);

    auto msa_tensor = prepare_msa_tensor(sequence);
    std::cout << "MSA Tensor:\n" << msa_tensor << std::endl;

    auto msa_embeddings = model.forward(msa_tensor);
    std::cout << "MSA Embeddings:\n" << msa_embeddings << std::endl;

    return 0;
}