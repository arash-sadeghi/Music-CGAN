from CONST_VARS import CONST
import matplotlib.pyplot as plt
torch = CONST.torch

def draw_hist(tensor, output_url):
    # Filter out zero values
    non_zero_tensor = tensor[tensor != 0]

    if non_zero_tensor.numel() == 0:
        print("The tensor has no non-zero values to visualize.")
        return

    num_bins = 100  

    # Find the minimum and maximum values in the non-zero tensor
    min_val = torch.min(non_zero_tensor).item()
    max_val = torch.max(non_zero_tensor).item()

    # Define the bins
    bins = torch.linspace(min_val, max_val, num_bins + 1)

    # Flatten the non-zero tensor
    flat_tensor = non_zero_tensor.flatten()

    # Count the members in each bin
    hist = torch.histc(flat_tensor.float(), bins=num_bins, min=min_val, max=max_val)

    # Find the range with the most members
    max_bin_index = torch.argmax(hist).item()
    range_with_most_members = (bins[max_bin_index].item(), bins[max_bin_index + 1].item())
    plt.figure()
    # Plot the histogram
    plt.bar(bins[:-1].cpu().numpy(), hist.cpu().numpy(), width=(bins[1] - bins[0]).item(), align='edge', alpha=0.7, color='blue')
    plt.axvspan(range_with_most_members[0], range_with_most_members[1], color='orange', alpha=0.3, label="Most Members Range")

    # Add labels and title
    plt.title("Histogram of Non-Zero Tensor Values")
    plt.xlabel("Value Range")
    plt.ylabel("Count")
    plt.legend()

    # Save the plot to the specified output URL
    plt.savefig(output_url)
    plt.close()

    print(f"Range with the most members: {range_with_most_members} (Count: {hist[max_bin_index].item()})")
