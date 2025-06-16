source .venv/bin/activate

# The below is necessary to ensure that the NVIDIA GPU is used for rendering
# This may be required for laptop users with hybrid graphics setups
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
