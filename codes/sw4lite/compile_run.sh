#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX environment variable not set"
  exit 1
fi  

"${CLANG_PREFIX}/bin/clang++" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  -mllvm -polly-position=early -O3 -march=native \
  *.C -o "${BASENAME}" \
  -I"/usr/include/x86_64-linux-gnu/mpich" -lmpich -llapack -lm \
  -I"${SCRIPTPATH}" -DSW4_CROUTINES -DNDEBUG -mllvm -polly -mllvm -polly-position=early -mllvm -polly-only-func=ljForce_kernel -mllvm -polly-allow-nonaffine -mllvm -polly-process-unprofitable -mllvm -polly-allow-nonaffine-branches -mllvm -polly-print-instructions -mllvm -polly-use-llvm-names

"${SCRIPTPATH}/${BASENAME}" LOH.1-h100.in
