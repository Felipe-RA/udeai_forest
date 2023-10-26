# Use the ROCm PyTorch image as the base image
FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1

# Update pip
RUN pip install --upgrade pip

# Install Jupyter, pandas, scikit-learn, and other common data science libraries
RUN pip install jupyter pandas scikit-learn matplotlib seaborn numpy

# Expose the port Jupyter Notebook will run on
EXPOSE 8888

# Set the working directory
WORKDIR /mnt

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
