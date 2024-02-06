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
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)