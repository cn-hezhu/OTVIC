from pathlib import Path

import setuptools

package_dir = "python-sdk"
setup_dir = Path(__file__).parent / "setup"

# Since nuScenes 2.0 the requirements are stored in separate files.
requirements = []
for req_path in (setup_dir / "requirements.txt").read_text().splitlines():
    if req_path.startswith("#"):
        continue
    req_path = req_path.replace("-r ", "")
    requirements += (setup_dir / req_path).read_text().splitlines()


setuptools.setup(
    name="nuscenes-devkit",
    version="1.1.9",
    author="Holger Caesar, Oscar Beijbom, Qiang Xu, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, "
    "Sergi Widjaja, Kiwoo Shin, Caglayan Dicle, Freddy Boulton, Whye Kit Fong, Asha Asvathaman, Lubing Zhou "
    "et al.",
    author_email="nuscenes@motional.com",
    description="The official devkit of the nuScenes dataset (www.nuscenes.org).",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/nutonomy/nuscenes-devkit",
    python_requires=">=3.6",
    install_requires=requirements,
    packages=setuptools.find_packages(package_dir),
    package_dir={"": package_dir},
    package_data={"": ["*.json"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="cc-by-nc-sa-4.0",
)
