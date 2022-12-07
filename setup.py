# This setup file is based on the one found here:
# https://towardsdatascience.com/create-your-custom-python-package-that-you-can-pip-install-from-your-git-repository-f90465867893

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pi_cai_reader_core',
    version='0.1.0',
    author='I.N.Tzortzis',
    author_email='i.n.tzortzis@gmail.com',
    description='PI-CAI_reader installation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/itzortzis/PI-CAI_reader',
    project_urls = {
        "Code": "https://github.com/itzortzis/PI-CAI_reader",
        "Bug Tracker": "https://github.com/itzortzis/PI-CAI_reader/issues"
    },
    license='GPL-3.0',
    packages=['pi_cai_reader_core'],
    install_requires=['numpy', 'torch', 'cv2', 'medpy.io', 'matplotlib.pyplot', 'os'],
)
