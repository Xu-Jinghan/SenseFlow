#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Simple setup script"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

abspath = os.path.dirname(os.path.realpath(__file__))
cuda_extra_compile_args = {}

if os.name == "nt":
    # CUDA 11.8 rejects the newest VS 2022 host toolset version by default.
    # Allow the local Windows build to proceed so the project uses the CUDA
    # quantization extension instead of falling back to the Python path.
    cuda_extra_compile_args["nvcc"] = [
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
    ]

with open(os.path.join(abspath, "requirements.txt")) as f:
    requirements = f.read().splitlines()

license_header = """#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""

# Generate version file
with open(os.path.join(abspath, "VERSION")) as f:
    version = f.read().strip()
with open(os.path.join(abspath, "pytorch_quantization/version.py"), "w") as f:
    f.write(license_header)
    f.write(F"__version__ = \"{version}\"")

setup(
    name="pytorch_quantization",
    version=version,
    description="NVIDIA Pytorch quantization toolkit",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],

    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pytorch_quantization.cuda_ext",
            sources=[os.path.join(abspath, "src/tensor_quant.cpp"),
                     os.path.join(abspath, "src/tensor_quant_gpu.cu")],
            extra_compile_args=cuda_extra_compile_args)
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    zip_safe=False,

    long_description=open("README.md", "r", encoding="utf-8").read(),
    url="https://github.com/nvidia/tensorrt/tools/pytorch-quantization",
    author="NVIDIA",
    author_email="nvidia@nvidia.com",
    license="Apache 2.0",
)
