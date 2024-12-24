import torch

from tnibs.utils.array import to_permutation

def _grad(out, inp):
    return torch.autograd.grad(
        out,
        inp,
        grad_outputs=torch.ones_like(inp),
        create_graph=True,
        allow_unused=True,
    )[0]



def describe_tensor(tensor, feature_dims):
    if isinstance(feature_dims, slice):
        feature_dims = list(range(len(tensor.shape))[feature_dims])
    else:
        feature_dims = [
            idx if idx >= 0 else len(tensor.shape) + idx for idx in feature_dims
        ]

    complement_dims = [i for i in range(tensor.dim()) if i not in feature_dims]

    min_value = torch.permute(tensor.amin(dim=complement_dims), to_permutation(feature_dims))

    max_value = torch.permute(tensor.amax(dim=complement_dims), to_permutation(feature_dims))
    mean_value = torch.permute(tensor.mean(dim=complement_dims), to_permutation(feature_dims))
    std_value = torch.permute(tensor.std(dim=complement_dims), to_permutation(feature_dims))

    # Print the results
    print(f"\nfeature_dims {feature_dims}: ")
    print(f"Type: {tensor.dtype}")
    print("Min:")
    print(min_value)
    print("Max:")
    print(max_value)
    print("Mean:")
    print(mean_value)
    print("Std:")
    print(std_value)
