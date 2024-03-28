import torch



# Change tensor's file name

X_name = "X_tensor.pth"
y_name = "y_tensor.pth"


# Load the tensors from disk

x_tensor = torch.load(X_name)
y_tensor = torch.load(y_name)

# Print the shapes and data types of the tensors
print("Shape of X_tensor:", x_tensor.shape)
print("Data type of X_tensor:", x_tensor.dtype)
print("Number of elements in X_tensor:", torch.numel(x_tensor))

print("Shape of y_tensor:", y_tensor.shape)
print("Data type of y_tensor:", y_tensor.dtype)
print("Number of elements in y_tensor:", torch.numel(y_tensor))

# Calculate the size in bytes
print("Size of X_tensor in bytes:", x_tensor.element_size() * x_tensor.nelement())
print("Size of y_tensor in bytes:", y_tensor.element_size() * y_tensor.nelement())
