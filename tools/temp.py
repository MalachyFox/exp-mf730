import torch

# Path to the file
file_path = 'embeddings/egemaps_embeddings.pt'

# Load the PyTorch file
data = torch.load(file_path)

# Check if the data is a list
if isinstance(data, list):
    lengths = []
    print(f'len data: {len(data)}')

    # Iterate through the list to extract lengths (L dimension)
    for i, tensor in enumerate(data):
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 3:
            lengths.append(tensor.shape[1])  # Extract L (2nd dimension)
        else:
            print(f"Item {i} is of unexpected type or shape: {type(tensor)} {tensor.shape if isinstance(tensor, torch.Tensor) else ''}")

    # Calculate statistics if lengths are found
    if lengths:
        mean_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        min_length = min(lengths)

        print(f"Mean Length (L): {mean_length}")
        print(f"Max Length (L): {max_length}")
        print(f"Min Length (L): {min_length}")
    else:
        print("No valid tensors with (1, L, 768) found in the list.")
else:
    print(f"Data is not a list. Found type: {type(data)}")
