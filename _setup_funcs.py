
import os, tempfile, subprocess, shutil, sys

# see http://openmp.org/wp/openmp-compilers/
# original OMP test taken from: https://stackoverflow.com/a/16555458

omp_test = \
r"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
"""

def check_for_openmp():
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    compiler = os.getenv('CC')
    if compiler is None:
        compiler = "gcc"
    openmp = '/openmp' if "win" == sys.platform[:3] else '-fopenmp'

    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(omp_test)

    with open(os.devnull, 'w') as fnull:
        result = subprocess.call([compiler, "-o test -v", openmp, filename],
                                 stdout=fnull, stderr=fnull)

    os.chdir(curdir)
    shutil.rmtree(tmpdir)

    return result == 0

if __name__ == "__main__":
    print(check_for_openmp())
