from setuptools import setup
setup( name = 'Signal_Analysis',
      version=' 0.1.3 ',
      description = 'Determines different characteristics of signals.',
      long_description = '',
      author = 'Brooke V Mosby',
      author_email = 'brooke.mosby.byu@gmail.com',
      url='https://github.com/brookemosby/Speech_Analysis',
      license='MIT',
      setup_requires = [ 'pytest-runner' ],
      tests_require = [ 'pytest', 'python-coveralls' ],
      install_requires = [
          "numpy",
          "peakutils"
      ],
      packages = [ 'Signal_Analysis' ] ,
      include_package_data = True,
      scripts = [ 'Signal_Analysis/Signal_Analysis.py', 'Signal_Analysis/sig_tools.py' ],
              
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Other Audience',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
      ],
)