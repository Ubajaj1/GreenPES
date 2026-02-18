from setuptools import setup, find_packages

setup(
    name="greenprompt",
    version="0.1.0",
    description="GreenPES: Green Prompt Efficiency Score for LLM optimization",
    author="Your Name",
    packages=find_packages(exclude=["tests", "experiments", "paper", "results"]),
    install_requires=[
        "tiktoken>=0.5.0",
    ],
    extras_require={
        "gemini": ["google-generativeai>=0.3.0"],
        "groq": ["groq>=0.4.0"],
        "all": ["google-generativeai>=0.3.0", "groq>=0.4.0"],
        "dev": ["pytest", "pandas", "scipy", "matplotlib"],
    },
    python_requires=">=3.9",
)
