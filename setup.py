from setuptools import setup, find_packages

setup(
    name='aipal_industrial_dqn',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'matplotlib',
        'stable-baselines3',
        'torch',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'train-industrial=aipal_industrial_dqn.trainsb3:main',
            'test-industrial=aipal_industrial_dqn.testsb3:main',
        ]
    },
)
