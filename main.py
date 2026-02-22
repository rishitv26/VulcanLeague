#
# Main file that runs everything. Run this in server.
#

# depending on version and computer, program may take some time to start
print("Loading...")

import util
import install
try:
    import run
except e as ModuleFoundError:
    print(e)
    install.main()

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA (NVIDIA GPU) is available.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple Silicion GPU) is available.")
else:
    device = torch.device("mps")
    print("Only CPU is available.")

print(f"Using device: {device}")

def main():
    if util.is_installed():
        run.main()
    else:
        install.main()

main()
                          
