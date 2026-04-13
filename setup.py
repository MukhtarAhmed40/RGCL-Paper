from setuptools import setup, find_packages

setup(
    name='rgcl',
    version='1.0.0',
    author='Anonymous',
    description='Robust Graph Contrastive Learning for Adversarial Network Intrusion Detection',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torch-geometric>=2.3.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
    ],
    python_requires='>=3.8',
)
