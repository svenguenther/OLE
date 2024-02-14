from setuptools import setup

setup(
    name='OLE',
    version='0.1',    
    description='Online Learning Emulator',
    url='https://github.com/svenguenther/OLE',
    author='Sven Günther',
    author_email='sven.guenther@rwth-aachen.de',
    license='...',
    packages=['OLE'],
    install_requires=[
                    'numpy',
                    'jax>=0.4.23',
                    'jaxlib>=0.4.23',
                    'tqdm',
                    'gpjax>=0.8.0',
                    'fasteners',
                    'emcee',
                    'python>=3.10',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ],
)