from setuptools import setup, find_packages

setup(
    name="eegspeech",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy<2.0.0",
        "matplotlib",
        "torch",
        "scikit-learn",
        "streamlit",
        "plotly",
        "scipy",
    ],
    entry_points={
        'console_scripts': [
            'eegspeech=eegspeech.app.cli:main',
        ],
    },
)
