from setuptools import setup, find_packages

setup(
    name="heart-seg-app",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "hsa = heart_seg_app.cli:main"
        ]
    },
)
