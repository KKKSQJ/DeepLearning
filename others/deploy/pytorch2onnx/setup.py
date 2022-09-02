from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='my_add',
    ext_modules=[cpp_extension.CppExtension('my_lib', ['my_add.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
