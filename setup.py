from setuptools import setup, find_packages

setup(
    name="orbslam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "scikit-learn",
        "matplotlib",
    ],
    author="Nitin Thakkar",
    author_email="thakkarnitin1998@gmail.com",
    description="A custom implementation of ORB-SLAM, a monocular SLAM system",
    python_requires=">=3.6",
)
