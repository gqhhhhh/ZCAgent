from setuptools import setup, find_packages

setup(
    name="zcagent",
    version="0.1.0",
    description="基于LLM+RAG的智能座舱一体化语义Agent系统",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "rank-bm25>=0.2.2",
        "pyyaml>=6.0",
        "jieba>=0.42.1",
        "langchain>=0.3.25",
        "langchain-core>=0.3.81",
        "langchain-openai>=0.3.0",
        "langgraph>=0.3.0",
        "pyautogen>=0.2.0",
        "mcp>=1.0.0",
    ],
)
